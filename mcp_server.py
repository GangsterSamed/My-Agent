#!/usr/bin/env python3
"""
MCP‑сервер для управления браузером (Playwright).
Видимый браузер, persistent session через storage state, без хардкода селекторов.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.types import ServerCapabilities, ToolsCapability
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from playwright.async_api import async_playwright

load_dotenv()

# Путь к storage state для persistent session (по умолчанию — сохраняем всегда)
STORAGE_STATE_PATH = (os.getenv("AGENT_STORAGE_STATE") or "browser_state.json").strip()
HEADLESS = os.getenv("AGENT_HEADLESS", "false").lower() == "true"
QUIET = os.getenv("AGENT_QUIET", "true").lower() in ("1", "true", "yes")
# Таймаут клика/ввода (мс). В click_element используем no_wait_after=True, чтобы после клика по ссылке
# Playwright не ждал навигацию до 30 с — иначе в фоне остаётся «Future exception was never retrieved».
ACTION_TIMEOUT_MS = 8_000


def _normalize_text(s: str | None) -> str:
    """Нормализация текста для Playwright: Unicode-пробелы и переносы → обычный пробел, схлопывание пробелов."""
    if s is None or not isinstance(s, str):
        return ""
    t = (
        s.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")  # thin space (п.4)
        .replace("\u2028", " ")
        .replace("\u2029", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    return " ".join(t.split())


def _tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="navigate",
            description="Перейти по URL. Вызови первым, если страница ещё не открыта.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL для перехода"}},
                "required": ["url"],
            },
        ),
        types.Tool(
            name="get_page_content",
            description="Получить содержимое текущей страницы: текст, кнопки, ссылки, поля ввода. В начале текста — список полей для ввода (field_index 1, 2, 3…) для type_text. Используй для анализа перед действиями.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_html": {
                        "type": "boolean",
                        "description": "Включать сырой HTML (редко нужно)",
                        "default": False,
                    }
                },
            },
        ),
        types.Tool(
            name="click_element",
            description="Кликнуть по элементу. Указывай text (видимый текст), или selector, или role. В диалоге scope автоматический — передавай text или selector элемента, не [role=dialog].",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Видимый текст элемента"},
                    "selector": {"type": "string", "description": "CSS-селектор элемента (не контейнера диалога)"},
                    "role": {"type": "string", "description": "ARIA-роль (button, link и т.д.)"},
                    "exact": {"type": "boolean", "description": "Точное совпадение текста", "default": False},
                },
            },
        ),
        types.Tool(
            name="type_text",
            description="Ввести текст в поле. Если полей несколько — укажи field_index (1, 2, 3…) по порядку полей в get_page_content, либо placeholder/selector. Длинный текст вводится в textarea. Checkbox/radio не поддерживаются.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Текст для ввода"},
                    "selector": {"type": "string", "description": "CSS-селектор поля"},
                    "placeholder": {"type": "string", "description": "Placeholder поля для поиска"},
                    "field_index": {
                        "type": "integer",
                        "description": "Номер поля по порядку (1-based). Порядок — как в DOM: все видимые input (кроме checkbox/radio), textarea, contenteditable. Используй, когда полей несколько и нужно заполнить конкретное по счёту.",
                        "minimum": 1,
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="scroll",
            description="Проскроллить страницу или внутренний контейнер. Без container_selector — скролл окна. С container_selector — скролл указанного элемента (меню, сайдбар и т.д.). Возвращает обновлённое содержимое страницы.",
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"], "default": "down"},
                    "amount": {"type": "integer", "description": "Пиксели", "default": 500},
                    "container_selector": {
                        "type": "string",
                        "description": "CSS-селектор скроллируемого контейнера (опционально). Если кнопки/ссылки внутри меню не видны — укажи контейнер из get_page_content.",
                    },
                },
            },
        ),
        types.Tool(
            name="go_back",
            description="Вернуться на предыдущую страницу в истории. Используй после перехода по ссылке, чтобы открыть следующую.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="extract_elements",
            description="Извлечь элементы по типу: links, buttons, inputs, headings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "element_type": {
                        "type": "string",
                        "enum": ["links", "buttons", "inputs", "headings"],
                        "default": "links",
                    }
                },
            },
        ),
        types.Tool(
            name="save_session",
            description="Сохранить сессию браузера (логины, cookies) в файл. После перезапуска агента сессия подхватится, вводить логин/пароль снова не нужно.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


class BrowserMCPServer:
    def __init__(self) -> None:
        self._server = Server("browser-mcp-server")
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def _list_tools() -> types.ListToolsResult:
            return types.ListToolsResult(tools=_tools())

        @self._server.call_tool(validate_input=False)
        async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            impl = {
                "navigate": self._navigate,
                "get_page_content": self._get_page_content,
                "click_element": self._click_element,
                "type_text": self._type_text,
                "scroll": self._scroll,
                "go_back": self._go_back,
                "extract_elements": self._extract_elements,
                "save_session": self._save_session,
            }
            fn = impl.get(name)
            if not fn:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))],
                    isError=True,
                )
            try:
                result = await fn(arguments)
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
                )
            except Exception as e:  # noqa: BLE001
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}, ensure_ascii=False))],
                    isError=True,
                )

    async def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        pw = await async_playwright().start()
        self._playwright = pw
        win = (os.getenv("AGENT_WINDOW_SIZE") or "1100x700").strip().lower()
        try:
            w, h = win.split("x", 1)
            win_w, win_h = max(800, int(w)), max(600, int(h))
        except Exception:
            win_w, win_h = 1100, 700
        opts: dict[str, Any] = {
            "headless": HEADLESS,
            "args": ["--window-size=%d,%d" % (win_w, win_h)] if not HEADLESS else [],
        }
        self._browser = await pw.chromium.launch(**opts)
        ctx_opts: dict[str, Any] = {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        if HEADLESS:
            ctx_opts["viewport"] = {"width": 1920, "height": 1080}
        else:
            ctx_opts["viewport"] = None
        path = Path(STORAGE_STATE_PATH).resolve()
        if path.exists():
            ctx_opts["storage_state"] = str(path)
            if not QUIET:
                print("Сессия загружена: %s" % path, file=sys.stderr)
        self._context = await self._browser.new_context(**ctx_opts)
        self._page = await self._context.new_page()
        await self._page.add_init_script(
            "document.querySelectorAll('a[target=\"_blank\"], a[target=\"_new\"]').forEach(a => a.removeAttribute('target'));"
        )
        if not QUIET:
            print("Браузер запущен (headless=%s)" % HEADLESS, file=sys.stderr)

    async def _ensure_single_tab(self) -> None:
        """Закрыть лишние вкладки, оставить одну — агент работает только с одной вкладкой."""
        pages = self._context.pages
        if len(pages) <= 1:
            return
        for p in pages[1:]:
            await p.close()
        self._page = pages[0]

    def _strip_dialog_selector(self, sel: str) -> str | None:
        """Убрать ведущий [role=dialog/alertdialog] / [aria-modal] из селектора. Если после этого пусто — None (использовать text)."""
        if not sel or not isinstance(sel, str):
            return None
        s = sel.strip()
        m = re.match(
            r"^\s*(?:\[role\s*=\s*[\"'](?:dialog|alertdialog)[\"']\]|\[aria-modal\s*=\s*[\"']true[\"']\])\s*(.*)$",
            s,
            re.IGNORECASE,
        )
        if not m:
            return s or None
        rest = re.sub(r"^[\s>+~]+", "", m.group(1).strip())
        return rest if rest else None

    async def _get_dialog_locator(self):
        """Первый видимый ARIA-диалог: role=dialog | role=alertdialog | aria-modal=true. Без классовых селекторов."""
        batches = [
            "[role=\"dialog\"], [role=\"alertdialog\"], [aria-modal=\"true\"]",
        ]
        for sel in batches:
            loc = self._page.locator(sel)
            n = await loc.count()
            for i in range(n):
                el = loc.nth(i)
                try:
                    if await el.is_visible():
                        return el
                except Exception:
                    continue
        return None

    async def _navigate(self, args: dict[str, Any]) -> dict[str, Any]:
        url = (args.get("url") or "").strip()
        if not url:
            return {"success": False, "error": "Укажи url для перехода."}
        await self._ensure_browser()
        await self._ensure_single_tab()
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}
        return {
            "success": True,
            "url": self._page.url,
            "title": await self._page.title(),
            "page_navigated": True,
        }

    _PAGE_CONTENT_SCRIPT = """
        () => {
            document.querySelectorAll('a[target="_blank"], a[target="_new"]').forEach(a => a.removeAttribute('target'));
            const root = document.body;
            let bodyText = root?.innerText ?? '';
            const sel = (s, r) => (r || root).querySelectorAll(s);
            const arr = (q) => Array.from(q);
            const vw = window.innerWidth || 1280, vh = window.innerHeight || 720;
            const isInViewport = (el) => {
                if (!el || !el.getBoundingClientRect) return false;
                const r = el.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) return false;
                return r.top < vh && r.bottom > 0 && r.left < vw && r.right > 0;
            };
            const isOffScreenOrSrOnly = (el) => {
                if (!el || !el.getBoundingClientRect) return true;
                const r = el.getBoundingClientRect();
                if (r.left < -500 || r.top < -500 || r.left > vw + 500 || r.top > vh + 500) return true;
                const s = window.getComputedStyle ? window.getComputedStyle(el) : {};
                const leftVal = (s.left || '').trim();
                if (leftVal.indexOf('-9999') !== -1 || (s.clip && s.clip.indexOf('rect(0px') === 0)) return true;
                const cls = (el.className || '').toLowerCase();
                if (/sr-only|visually-hidden|skip-link|off-screen|screen-reader/.test(cls)) return true;
                return false;
            };
            let modal = null;
            const dialogEl = root.querySelector('[role="dialog"]') || root.querySelector('[role="alertdialog"]') || root.querySelector('[aria-modal="true"]');
            let dialog = null;
            if (dialogEl) {
                const r = dialogEl.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) { dialog = dialogEl; }
            }
            if (dialog) {
                const mt = (dialog.innerText || '').trim().slice(0, 4000);
                const mb = arr(sel('button, [role="button"], input[type="submit"]', dialog))
                    .map(el => el.innerText?.trim() || el.value || el.getAttribute('aria-label') || '')
                    .filter(t => t).slice(0, 20);
                const mi = arr(sel('input:not([type=hidden]):not([type=checkbox]):not([type=radio]), textarea, select', dialog))
                    .map(el => ({ type: el.type || el.tagName.toLowerCase(), placeholder: el.placeholder || '', name: el.name || '', id: el.id || '' }))
                    .slice(0, 20);
                modal = { text: mt, buttons: mb, inputs: mi };
            }
            const allButtonEls = arr(sel('button, [role="button"], input[type="submit"]'));
            const visibleButtonEls = allButtonEls.filter(el => isInViewport(el) && !isOffScreenOrSrOnly(el));
            const buttons = visibleButtonEls
                .map(el => el.innerText?.trim() || el.value || el.getAttribute('aria-label') || '')
                .filter(t => t).slice(0, 30);
            const allLinkEls = arr(sel('a[href]'));
            const visibleLinkEls = allLinkEls.filter(el => isInViewport(el) && !isOffScreenOrSrOnly(el));
            const links = visibleLinkEls
                .map(el => ({ text: (el.innerText || '').trim().slice(0, 200), href: el.href }))
                .filter(l => l.text).slice(0, 50);
            const filtered_buttons = allButtonEls.length - visibleButtonEls.length;
            const filtered_links = allLinkEls.length - visibleLinkEls.length;
            const hiddenClickableTexts = allButtonEls
                .filter(el => !isInViewport(el) || isOffScreenOrSrOnly(el))
                .map(el => (el.innerText || el.value || el.getAttribute('aria-label') || '').trim())
                .filter(t => t.length > 0);
            const hiddenLinkTexts = allLinkEls
                .filter(el => !isInViewport(el) || isOffScreenOrSrOnly(el))
                .map(el => (el.innerText || '').trim())
                .filter(t => t.length > 0);
            const hiddenTextsToMask = [...new Set([...hiddenClickableTexts, ...hiddenLinkTexts])].filter(t => t.length < 300);
            hiddenTextsToMask.forEach(t => {
                if (bodyText.indexOf(t) !== -1) bodyText = bodyText.replace(t, '[скрытая ссылка]');
            });
            const inputs = arr(sel('input, textarea, select'))
                .map(el => ({ type: el.type, placeholder: el.placeholder || '', name: el.name || '', id: el.id || '' }))
                .slice(0, 30);
            const isVisible = (el) => {
                if (!el || !el.getBoundingClientRect) return false;
                const r = el.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) return false;
                const s = window.getComputedStyle ? window.getComputedStyle(el) : {};
                if (s.display === 'none' || s.visibility === 'hidden') return false;
                return true;
            };
            const fillableSelector = 'input:not([type=checkbox]):not([type=radio]):not([type=hidden]), textarea, [contenteditable="true"]';
            const fillableInputs = arr(sel(fillableSelector))
                .filter(isVisible)
                .slice(0, 50)
                .map((el, i) => {
                    const ph = (el.placeholder || '').trim();
                    const name = (el.name || '').trim();
                    const id = (el.id || '').trim();
                    const type = (el.type || el.tagName.toLowerCase() || '').toLowerCase();
                    let label = '';
                    if (el.id && root) {
                        const arrLabels = arr(root.querySelectorAll('label'));
                        const labelEl = arrLabels.find(function(lab) { return lab.htmlFor === el.id; });
                        if (labelEl) label = (labelEl.innerText || '').trim().slice(0, 80);
                    }
                    if (!label && el.closest('label')) {
                        const par = el.closest('label');
                        if (par) label = (par.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                    }
                    return { index: i + 1, placeholder: ph, name, id, type, label: label.slice(0, 80) };
                });
            const maxText = 18000;
            let text = bodyText.slice(0, maxText);
            if (modal && modal.text) {
                let prefix = '\\n[Модальное окно]\\n';
                if (modal.buttons && modal.buttons.length) {
                    const btns = modal.buttons.map(function(t) {
                        return (t || '').trim().replace(/\\s+/g, ' ').slice(0, 120);
                    }).filter(Boolean);
                    if (btns.length)
                        prefix += 'Кнопки в модалке: ' + btns.map(function(t) { return '«' + t + '»'; }).join(', ') + '\\n\\n';
                }
                if (modal.inputs && modal.inputs.length) {
                    const parts = modal.inputs.map(function(inp) {
                        var ph = (inp.placeholder || '').trim().slice(0, 80);
                        return ph ? ('placeholder «' + ph + '»') : (inp.name || inp.id || inp.type || '');
                    }).filter(Boolean);
                    if (parts.length)
                        prefix += 'Поля в модалке: ' + parts.join(', ') + '\\n\\n';
                }
                prefix += modal.text.slice(0, 2500) + '\\n\\n';
                text = prefix + text.slice(0, maxText - prefix.length);
            }
            const scrollable = [];
            const check = (el) => {
                if (!el || el === document.body) return;
                const s = getComputedStyle(el);
                const oy = (s.overflowY || s.overflow || '').toLowerCase();
                if (!/auto|scroll|overlay/.test(oy) || el.scrollHeight <= el.clientHeight) return;
                let sel = el.id ? '#' + el.id : null;
                if (!sel && el.className && typeof el.className === 'string') {
                    const c = (el.className.trim().split(/\\s+/)[0] || '').replace(/[^a-zA-Z0-9_-]/g, '');
                    if (c) sel = el.tagName.toLowerCase() + '.' + c;
                }
                if (!sel) sel = el.getAttribute('role') ? '[role="' + el.getAttribute('role') + '"]' : el.tagName.toLowerCase();
                if (sel) scrollable.push(sel);
            };
            arr(sel('main, [role="main"], [role="feed"], aside, [class*="menu"], [class*="list"], [class*="scroll"]')).slice(0, 8).forEach(check);
            if (scrollable.length) {
                text = 'Скроллируемые контейнеры (для scroll с container_selector): ' + [...new Set(scrollable)].slice(0, 5).join(', ') + '\\n\\n' + text;
            }
            if (fillableInputs.length > 0) {
                const parts = fillableInputs.map(function(inp) {
                    const ph = inp.placeholder ? ('placeholder «' + inp.placeholder + '»') : '';
                    const nm = inp.name ? ('name «' + inp.name + '»') : '';
                    const lb = inp.label ? ('label «' + inp.label + '»') : '';
                    const desc = [ph, nm, lb].filter(Boolean).join(', ') || ('type ' + inp.type);
                    return inp.index + ' — ' + desc;
                });
                text = 'Поля для ввода (field_index для type_text): ' + parts.join('; ') + '\\n\\n' + text;
            }
            return {
                text,
                modal,
                buttons,
                links,
                inputs,
                fillable_inputs: fillableInputs,
                scrollable_containers: [...new Set(scrollable)].slice(0, 5),
                url: window.location.href,
                title: document.title,
                filtered_buttons: filtered_buttons,
                filtered_links: filtered_links
            };
        }
    """

    async def _fetch_page_content(self) -> dict[str, Any]:
        """Выполнить evaluate и вернуть {text, modal, buttons, links, inputs, url, title}."""
        return await self._page.evaluate(self._PAGE_CONTENT_SCRIPT)

    async def _get_page_content(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        include_html = bool(args.get("include_html"))
        try:
            content = await self._fetch_page_content()
            out: dict[str, Any] = {"success": True, "content": {k: v for k, v in content.items() if k not in ("filtered_buttons", "filtered_links")}}
            if include_html:
                out["html"] = (await self._page.content())[:50000]
            return out
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _click_element(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        sel = args.get("selector")
        text = args.get("text")
        role = args.get("role")
        exact = bool(args.get("exact", False))
        scope = await self._get_dialog_locator()
        if scope is None:
            scope = self._page
        else:
            try:
                await scope.scroll_into_view_if_needed(timeout=2_000)
            except Exception:
                pass
        use_sel = sel
        use_text = _normalize_text(text) if text else None
        if scope != self._page and sel:
            stripped = self._strip_dialog_selector(sel)
            if stripped is None:
                if text:
                    use_sel = None
                else:
                    return {
                        "success": False,
                        "error": "Селектор задаёт только контейнер диалога. Укажи text или селектор элемента.",
                        "suggestion": "В диалоге используй text или селектор элемента, не [role=dialog].",
                    }
            elif stripped is not None:
                use_sel = stripped
        try:
            if use_sel:
                loc = scope.locator(use_sel).first
            elif use_text:
                loc = scope.get_by_text(use_text, exact=exact).first
            elif role:
                loc = scope.locator(f"[role='{role}']").first
            else:
                return {"success": False, "error": "Укажи text, selector или role"}

            url_before_click = self._page.url
            if use_sel or role:
                await loc.click(timeout=ACTION_TIMEOUT_MS, no_wait_after=True)
            elif use_text:
                try:
                    await loc.click(timeout=ACTION_TIMEOUT_MS, no_wait_after=True)
                except Exception:
                    if scope != self._page:
                        loc = scope.get_by_role("button", name=use_text).first
                        await loc.click(timeout=ACTION_TIMEOUT_MS, no_wait_after=True)
                    else:
                        raise
            await asyncio.sleep(0.8)
            url_after = self._page.url
            page_navigated = url_after != url_before_click
            result: dict[str, Any] = {"success": True}
            if page_navigated:
                result["page_navigated"] = True
            return result
        except Exception as e:
            err = str(e)
            sug = "Попробуй другой способ (другой текст/селектор) или scroll перед кликом. Используй только текст кнопок/ссылок из последнего get_page_content; для длинных названий — короткий фрагмент (первые слова)."
            if scope != self._page and ("timeout" in err.lower() or "exceeded" in err.lower()):
                sug = "Диалог открыт: ищи элементы только внутри него. Элементы страницы под ним недоступны."
            return {"success": False, "error": err, "suggestion": sug}

    async def _type_text(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        text = args.get("text") or ""
        sel = args.get("selector")
        placeholder = _normalize_text(args.get("placeholder")) or None
        field_index_raw = args.get("field_index")
        scope = await self._get_dialog_locator()
        if scope is None:
            scope = self._page
        else:
            try:
                await scope.scroll_into_view_if_needed(timeout=2_000)
            except Exception:
                pass
        field_index: int | None = None
        if field_index_raw is not None:
            try:
                idx = int(field_index_raw)
                if idx < 1:
                    return {
                        "success": False,
                        "error": "field_index должен быть >= 1.",
                        "suggestion": "Нумерация полей с 1 по порядку из get_page_content (inputs).",
                    }
                field_index = idx
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": "field_index должен быть целым числом >= 1.",
                    "suggestion": "Укажи номер поля по порядку (1, 2, 3…).",
                }
        input_ok = "input:not([type=checkbox]):not([type=radio]):not([type=hidden]):visible"
        combined_selector = f"{input_ok}, textarea:visible, [contenteditable='true']"
        try:
            if sel:
                await scope.locator(sel).first.fill(text, timeout=ACTION_TIMEOUT_MS)
            elif placeholder:
                await scope.get_by_placeholder(placeholder).first.fill(text, timeout=ACTION_TIMEOUT_MS)
            elif field_index is not None:
                loc = scope.locator(combined_selector).nth(field_index - 1)
                await loc.fill(text, timeout=ACTION_TIMEOUT_MS)
            elif len(text) > 80:
                txt = scope.locator("textarea:visible")
                if await txt.count() > 0:
                    await txt.first.fill(text, timeout=ACTION_TIMEOUT_MS)
                else:
                    loc = scope.locator(
                        f"{input_ok}, [contenteditable='true']"
                    ).first
                    await loc.fill(text, timeout=ACTION_TIMEOUT_MS)
            else:
                loc = scope.locator(combined_selector).first
                await loc.fill(text, timeout=ACTION_TIMEOUT_MS)
            await asyncio.sleep(0.3)
            return {"success": True}
        except Exception as e:
            err = str(e)
            err_lower = err.lower()
            suf = " Вызови get_page_content и укажи placeholder, selector или field_index (1, 2, 3…) по порядку полей; для сопроводительного письма ищи textarea."
            if ("timeout" in err_lower or "exceeded" in err_lower) and (
                "input" in err_lower or "textarea" in err_lower or "placeholder" in err_lower or "locator(" in err_lower
            ):
                suf += " Если форма с полями ещё не открыта — сначала открой её (кнопка/ссылка на странице), затем get_page_content и заполняй поля по placeholder/selector/field_index. [hint: open_form_first]"
            if scope != self._page:
                suf += " Диалог открыт: используй placeholder или selector для поля внутри него."
            if field_index is not None:
                suf += " Если field_index — проверь, что номер не больше числа полей (в списке inputs не считай checkbox/radio)."
            suf += " Нумерация field_index — из строки «Поля для ввода» в начале get_page_content."
            return {"success": False, "error": err + suf}

    async def _scroll(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        direction = (args.get("direction") or "down").lower()
        amount = max(0, int(args.get("amount") or 500))
        delta = amount if direction == "down" else -amount
        container_sel = (args.get("container_selector") or "").strip()

        if container_sel:
            try:
                result = await self._page.evaluate(
                    """
                    ([selector, delta]) => {
                        const el = document.querySelector(selector);
                        if (!el) return { error: "Контейнер не найден: " + selector };
                        const before = el.scrollTop || 0;
                        el.scrollTop = Math.max(0, before + delta);
                        return {
                            scrollTop: el.scrollTop,
                            scrollHeight: el.scrollHeight,
                            clientHeight: el.clientHeight
                        };
                    }
                    """,
                    [container_sel, delta],
                )
                if isinstance(result, dict) and result.get("error"):
                    return {"success": False, "error": result["error"]}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            await self._page.evaluate("d => window.scrollBy(0, d)", delta)

        await asyncio.sleep(0.4)
        out: dict[str, Any] = {"success": True}
        try:
            content = await self._fetch_page_content()
            out["content"] = content
        except Exception:
            pass
        return out

    async def _go_back(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        try:
            await self._page.go_back(wait_until="domcontentloaded", timeout=15_000)
        except Exception as e:
            return {"success": False, "error": str(e)}
        return {
            "success": True,
            "url": self._page.url,
            "title": await self._page.title(),
        }

    async def _extract_elements(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        kind = args.get("element_type") or "links"
        selectors = {
            "links": "a[href]",
            "buttons": "button, [role='button'], input[type='submit']",
            "inputs": "input, textarea, select",
            "headings": "h1, h2, h3, h4, h5, h6",
        }
        sel = selectors.get(kind, "a")
        try:
            els = await self._page.evaluate(
                """
                (selector) => {
                    return Array.from(document.querySelectorAll(selector))
                        .map(el => ({
                            text: (el.innerText || el.value || '').trim().slice(0, 300),
                            tag: el.tagName,
                            id: el.id || '',
                            href: el.href || ''
                        }))
                        .filter(e => e.text);
                }
                """,
                sel,
            )
            return {"success": True, "element_type": kind, "elements": els}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _save_session(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        path = Path(STORAGE_STATE_PATH).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=str(path))
            if not QUIET:
                print("Сессия сохранена: %s" % path, file=sys.stderr)
            return {"success": True, "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="browser-mcp-server",
                    server_version="1.0.0",
                    capabilities=ServerCapabilities(tools=ToolsCapability(listChanged=False)),
                ),
            )

    async def cleanup(self) -> None:
        if self._context:
            path = Path(STORAGE_STATE_PATH).resolve()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                await self._context.storage_state(path=str(path))
                if not QUIET:
                    print("Сессия сохранена при выходе: %s" % path, file=sys.stderr)
            except Exception:
                pass
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


async def main() -> None:
    server = BrowserMCPServer()
    try:
        await server.run()
    except KeyboardInterrupt:
        pass
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
