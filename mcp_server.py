#!/usr/bin/env python3
"""MCP‑сервер управления браузером (Playwright). Persistent session через storage state."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

DIAG = os.getenv("AGENT_DIAG", "").strip().lower() in ("1", "true", "yes")

import mcp.types as types
from mcp.types import ServerCapabilities, ToolsCapability
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from playwright.async_api import async_playwright

load_dotenv()

STORAGE_STATE_PATH = (os.getenv("AGENT_STORAGE_STATE") or "browser_state.json").strip()
HEADLESS = os.getenv("AGENT_HEADLESS", "false").lower() == "true"
QUIET = os.getenv("AGENT_QUIET", "true").lower() in ("1", "true", "yes")
ACTION_TIMEOUT_MS = 8_000


def _normalize_text(s: str | None) -> str:
    """Unicode-пробелы и переносы → обычный пробел."""
    if s is None or not isinstance(s, str):
        return ""
    t = (
        s.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
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
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="click_element",
            description="Кликнуть по элементу. Указывай text (видимый текст) или selector. В диалоге scope автоматический — передавай text или selector элемента, не [role=dialog].",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Видимый текст элемента"},
                    "selector": {"type": "string", "description": "CSS-селектор элемента (не контейнера диалога)"},
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
            description="Проскроллить страницу или модалку. Без container_selector: если открыто модальное окно — скроллится модалка; иначе — окно. С container_selector — скролл указанного элемента. Возвращает обновлённое содержимое страницы.",
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"], "default": "down"},
                    "amount": {"type": "integer", "description": "Пиксели", "default": 500},
                    "container_selector": {
                        "type": "string",
                        "description": "CSS-селектор скроллируемого контейнера (опционально). Если кнопки/ссылки внутри списка или контейнера не видны — укажи контейнер из get_page_content.",
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
    ]


class BrowserMCPServer:
    def __init__(self) -> None:
        self._server = Server("browser-mcp-server")
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._last_nav_time: float | None = None
        self._last_click_time: float | None = None
        self._last_get_content_time: float | None = None
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
        """Оставить одну вкладку."""
        pages = self._context.pages
        if len(pages) <= 1:
            return
        for p in pages[1:]:
            await p.close()
        self._page = pages[0]

    def _strip_dialog_selector(self, sel: str) -> str | None:
        """Убрать ведущий [role=dialog] / [aria-modal] из селектора."""
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
        """Первый видимый ARIA-диалог."""
        sel = "[role=\"dialog\"], [role=\"alertdialog\"], [aria-modal=\"true\"]"
        loc = self._page.locator(sel)
        n = await loc.count()
        for i in range(n):
            el = loc.nth(i)
            try:
                if await el.is_visible():
                    return el
            except Exception:
                continue
        if DIAG:
            print("[DIAG] _get_dialog_locator no visible dialog", file=sys.stderr)
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
        self._last_nav_time = time.monotonic()
        return {
            "success": True,
            "url": self._page.url,
            "title": await self._page.title(),
            "page_navigated": True,
        }

    async def _diag_page(self) -> dict[str, Any]:
        try:
            return await self._page.evaluate(
                """
                () => {
                    const body = (document.body && document.body.innerText) || '';
                    const scrollY = window.scrollY || 0;
                    const scrollHeight = document.documentElement.scrollHeight || 0;
                    const buttons = Array.from(document.querySelectorAll('button, [role="button"], input[type="submit"]'));
                    const btexts = buttons.map(el => (el.innerText || el.value || el.getAttribute('aria-label') || '').trim()).filter(Boolean);
                    return {
                        scrollY,
                        scrollHeight,
                        bodyLen: body.length,
                        numButtons: buttons.length,
                        numInputs: document.querySelectorAll('input, textarea, select').length,
                        firstButtons: btexts.slice(0, 8)
                    };
                }
                """
            )
        except Exception as e:
            return {"error": str(e)}

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
            const selectableOptions = [];
            const radioCheckboxSel = 'input[type=radio], input[type=checkbox]';
            arr(sel(radioCheckboxSel)).filter(isVisible).slice(0, 50).forEach(function(el, idx) {
                let labelText = '';
                if (el.id && root) {
                    const arrLabels = arr(root.querySelectorAll('label'));
                    const labelEl = arrLabels.find(function(lab) { return lab.htmlFor === el.id; });
                    if (labelEl) labelText = (labelEl.innerText || labelEl.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                }
                if (!labelText && el.closest('label')) {
                    const par = el.closest('label');
                    if (par) labelText = (par.innerText || par.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                }
                if (!labelText && el.parentElement) {
                    const sib = arr(el.parentElement.childNodes).find(function(n) { return n.nodeType === 3 && (n.textContent || '').trim().length > 0; });
                    if (sib) labelText = (sib.textContent || '').trim().slice(0, 80);
                }
                if (labelText) {
                    const state = el.checked ? 'выбрано' : '';
                    selectableOptions.push({ type: el.type, label: labelText, checked: el.checked, state: state });
                }
            });
            arr(sel('select')).filter(isVisible).slice(0, 20).forEach(function(sel_el) {
                let labelText = '';
                if (sel_el.id && root) {
                    const arrLabels = arr(root.querySelectorAll('label'));
                    const labelEl = arrLabels.find(function(lab) { return lab.htmlFor === sel_el.id; });
                    if (labelEl) labelText = (labelEl.innerText || '').trim().slice(0, 60);
                }
                if (!labelText && sel_el.closest('label')) {
                    const par = sel_el.closest('label');
                    if (par) labelText = (par.innerText || '').trim().slice(0, 60);
                }
                const opts = arr(sel_el.querySelectorAll('option')).map(function(o) { return (o.textContent || '').trim(); }).filter(Boolean).slice(0, 10);
                if (opts.length > 0) {
                    const selectedVal = sel_el.value || '';
                    selectableOptions.push({ type: 'select', label: labelText || 'выбор', options: opts, selected: selectedVal });
                }
            });
            const maxText = 18000;
            let text = bodyText.slice(0, maxText);
            if (modal && modal.text) {
                let prefix = '\\n=== Модальное окно (действуй здесь) ===\\n';
                if (modal.buttons && modal.buttons.length) {
                    const btns = modal.buttons.map(function(t) {
                        return (t || '').trim().replace(/\\s+/g, ' ').slice(0, 120);
                    }).filter(Boolean);
                    if (btns.length)
                        prefix += 'Кнопки: ' + btns.map(function(t) { return '«' + t + '»'; }).join(', ') + '\\n\\n';
                }
                const modalRoot = document.querySelector('[role="dialog"]') || document.querySelector('[aria-modal="true"]');
                if (modalRoot) {
                    const modalSelectableOpts = [];
                    arr(modalRoot.querySelectorAll(radioCheckboxSel)).filter(isVisible).slice(0, 20).forEach(function(el) {
                        let labelText = '';
                        if (el.id) {
                            const labelEl = arr(modalRoot.querySelectorAll('label')).find(function(lab) { return lab.htmlFor === el.id; });
                            if (labelEl) labelText = (labelEl.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 60);
                        }
                        if (!labelText && el.closest('label')) labelText = (el.closest('label').innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 60);
                        if (!labelText && el.parentElement) {
                            const sib = arr(el.parentElement.childNodes).find(function(n) { return n.nodeType === 3 && (n.textContent || '').trim().length > 0; });
                            if (sib) labelText = (sib.textContent || '').trim().slice(0, 60);
                        }
                        if (labelText) {
                            const state = el.checked ? ' (выбрано)' : '';
                            modalSelectableOpts.push('[' + el.type + '] ' + labelText + state);
                        }
                    });
                    arr(modalRoot.querySelectorAll('select')).filter(isVisible).slice(0, 5).forEach(function(sel_el) {
                        const opts = arr(sel_el.querySelectorAll('option')).map(function(o) { return (o.textContent || '').trim(); }).filter(Boolean).slice(0, 8);
                        if (opts.length > 0) {
                            modalSelectableOpts.push('[select] варианты: ' + opts.join(', '));
                        }
                    });
                    if (modalSelectableOpts.length > 0) {
                        prefix += 'Опции (выбери одну, затем подтверди): ' + modalSelectableOpts.join('; ') + '\\n\\n';
                    }
                }
                if (modal.inputs && modal.inputs.length) {
                    const parts = modal.inputs.map(function(inp) {
                        var ph = (inp.placeholder || '').trim().slice(0, 80);
                        return ph ? ('placeholder «' + ph + '»') : (inp.name || inp.id || inp.type || '');
                    }).filter(Boolean);
                    if (parts.length)
                        prefix += 'Поля: ' + parts.join(', ') + '\\n\\n';
                }
                prefix += modal.text.slice(0, 2500) + '\\n\\n--- Ниже: страница (фон). scroll без container прокручивает модалку. ---\\n\\n';
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
            if (selectableOptions.length > 0 && !modal) {
                const parts = selectableOptions.map(function(opt) {
                    if (opt.type === 'select') {
                        const lbl = opt.label ? (opt.label + ': ') : '';
                        return '[select] ' + lbl + opt.options.slice(0, 6).join(', ');
                    } else {
                        const state = opt.checked ? ' (выбрано)' : '';
                        return '[' + opt.type + '] ' + opt.label + state;
                    }
                }).slice(0, 15);
                text = 'Опции для выбора (клик по подписи): ' + parts.join('; ') + '\\n\\n' + text;
            }
            return {
                text,
                modal,
                buttons,
                links,
                inputs,
                fillable_inputs: fillableInputs,
                selectable_options: selectableOptions,
                scrollable_containers: [...new Set(scrollable)].slice(0, 5),
                url: window.location.href,
                title: document.title,
                filtered_buttons: filtered_buttons,
                filtered_links: filtered_links
            };
        }
    """

    async def _fetch_page_content(self) -> dict[str, Any]:
        """evaluate PAGE_CONTENT_SCRIPT."""
        return await self._page.evaluate(self._PAGE_CONTENT_SCRIPT)

    async def _get_page_content(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        if DIAG:
            d = await self._diag_page()
            tn = self._last_nav_time
            tc = self._last_click_time
            tg = self._last_get_content_time
            now = time.monotonic()
            parts = [
                "get_page_content",
                "time_since_nav=%.1fs" % (now - tn) if tn is not None else "time_since_nav=N/A",
                "time_since_click=%.1fs" % (now - tc) if tc is not None else "time_since_click=N/A",
                "time_since_last_get=%.1fs" % (now - tg) if tg is not None else "time_since_last_get=N/A",
            ]
            if "error" in d:
                parts.append("diag_error=" + str(d["error"]))
            else:
                parts.extend([
                    "scrollY=%d" % d.get("scrollY", 0),
                    "scrollH=%d" % d.get("scrollHeight", 0),
                    "bodyLen=%d" % d.get("bodyLen", 0),
                    "numButtons=%d" % d.get("numButtons", 0),
                    "numInputs=%d" % d.get("numInputs", 0),
                    "firstBtns=%s" % (str(d.get("firstButtons") or [])[:120]),
                ])
            print("[DIAG] " + " ".join(str(p) for p in parts), file=sys.stderr)
        self._last_get_content_time = time.monotonic()
        try:
            content = await self._fetch_page_content()
            if DIAG:
                modal = content.get("modal")
                if modal:
                    mb = modal.get("buttons") or []
                    mt = (modal.get("text") or "")[:150]
                    print("[DIAG] get_page_content modal present buttons=%s modal_text_start=%r" % (
                        json.dumps(mb[:10], ensure_ascii=False)[:200],
                        mt.replace("\n", " ")[:100],
                    ), file=sys.stderr)
                btns = content.get("buttons") or []
                print("[DIAG] get_page_content page_buttons_first20=%s" % json.dumps([b[:40] for b in btns[:20]], ensure_ascii=False)[:500], file=sys.stderr)
            fb = content.get("filtered_buttons") or 0
            fl = content.get("filtered_links") or 0
            if fb > 0 or fl > 0:
                hint = "Вне viewport: %d кнопок, %d ссылок. Используй scroll если нужны элементы из списка.\n\n" % (fb, fl)
                content["text"] = hint + (content.get("text") or "")
            if DIAG and (fb > 0 or fl > 0):
                print(
                    "[DIAG] filter_off_screen filtered_buttons=%d filtered_links=%d (off-screen/sr-only excluded from content)"
                    % (fb, fl),
                    file=sys.stderr,
                )
            out: dict[str, Any] = {"success": True, "content": {k: v for k, v in content.items() if k not in ("filtered_buttons", "filtered_links")}}
            return out
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _click_element(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        sel = args.get("selector")
        text = args.get("text")
        if DIAG:
            tg = self._last_get_content_time
            now = time.monotonic()
            tsg = (now - tg) if tg is not None else None
            pre = "click_element start"
            if text:
                pre += " text=%r" % (text[:60] + "…" if len(text) > 60 else text)
            if sel:
                pre += " selector=%r" % (sel[:60] + "…" if len(sel) > 60 else sel)
            pre += " time_since_get_content=%s" % ("%.1fs" % tsg if tsg is not None else "N/A")
            print("[DIAG] " + pre, file=sys.stderr)
        _CHECK_DIALOG_SCRIPT = """
            () => {
                const d = document.querySelector('[role="dialog"]') || document.querySelector('[aria-modal="true"]');
                if (!d) return { hasDialog: false };
                const sel = 'button, [role="button"], a';
                const btns = Array.from(d.querySelectorAll(sel));
                const texts = btns.map(el => (el.innerText || el.textContent || el.getAttribute('aria-label') || '').trim()).filter(Boolean);
                const full = (d.innerText || d.textContent || '').trim().replace(/\\s+/g, ' ');
                const modalContext = full.split('\\n')[0].slice(0, 100);
                const modalFullText = full.slice(0, 2000).toLowerCase();
                const controlRe = /закрыть|отмена|cancel|dismiss|уменьшить|увеличить|decrease|increase|^\\s*[-+]\\s*$|\\bclose\\b|\\bminus\\b|\\bplus\\b/i;
                const nonControl = texts.filter(t => t && !controlRe.test((t || '').replace(/\\s+/g, ' ').trim()));
                const primaryActionBtn = nonControl.length > 0 ? nonControl[0] : null;
                return { hasDialog: true, btnCount: btns.length, btnTexts: texts.slice(0, 15), modalContext: modalContext || null, modalFullText, primaryActionBtn };
            }
        """
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
        effective_scope = scope
        if use_text and scope != self._page:
            try:
                diag_d = await self._page.evaluate(_CHECK_DIALOG_SCRIPT)
                if isinstance(diag_d, dict) and diag_d.get("hasDialog") and diag_d.get("btnTexts"):
                    btn_texts = diag_d.get("btnTexts", [])
                    search_lower = (use_text or "").lower().strip()
                    search_parts = [search_lower]
                    if len(search_lower) > 20:
                        words = search_lower.split()[:5]
                        search_parts.append(" ".join(words))
                        search_parts.append(" ".join(words[:3]))
                    in_dialog = False
                    for bt in btn_texts:
                        bt_norm = (bt or "").replace("\n", " ").replace("\r", " ").strip().lower()
                        if not bt_norm:
                            continue
                        for sp in search_parts:
                            if len(sp) >= 2 and (sp in bt_norm or bt_norm in sp or bt_norm.startswith(sp[:30]) or (len(sp) <= 60 and sp.startswith(bt_norm[:40]))):
                                in_dialog = True
                                break
                        if in_dialog:
                            break
                    if not in_dialog:
                        modal_full = (diag_d.get("modalFullText") or "").lower()
                        if modal_full and any(
                            len(sp) >= 2 and sp in modal_full
                            for sp in search_parts
                        ):
                            in_dialog = True
                            if DIAG:
                                print("[DIAG] click_element: text %r in modal body (not in btns) -> dialog scope" % (
                                    (use_text or "")[:50],
                                ), file=sys.stderr)
                    if not in_dialog:
                        btn_texts = diag_d.get("btnTexts") or []
                        btns_preview = ", ".join((t or "")[:30] for t in btn_texts[:6] if t)
                        if DIAG:
                            print("[DIAG] click_element: text %r not in dialog btnTexts=%s" % (
                                (use_text or "")[:50],
                                btns_preview[:80],
                            ), file=sys.stderr)
                        sug = "Проверь текст кнопок в модальном окне."
                        if btns_preview:
                            sug = f"Кнопки в диалоге: {btns_preview[:120]}. Выбери подходящую."
                        return {
                            "success": False,
                            "error": f"В модальном окне нет «{use_text}».",
                            "suggestion": sug,
                        }
            except Exception:
                pass
        if DIAG and use_text:
            esc = "page" if effective_scope == self._page else "dialog"
            print("[DIAG] click_element effective_scope=%s use_text=%r" % (esc, (use_text or "")[:60]), file=sys.stderr)
        if DIAG and text and any(c in text for c in "\xa0\u202f\u2009\u2028\u2029\n\r"):
            print("[DIAG] click_element text normalized (had unicode spaces/newlines)", file=sys.stderr)
        if DIAG and use_text and (" " in use_text or "\n" in use_text):
            print("[DIAG] click_element request has whitespace, collapsed match enabled", file=sys.stderr)
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
        _JS_CLICK_SCRIPT = """
            ([textArg]) => {
                const raw = (textArg || '').trim();
                if (!raw) return { ok: false };
                const text = raw.toLowerCase();
                const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                const collapseSpaces = (s) => (s || '').replace(/\\s+/g, '').toLowerCase();
                const firstWords = (s, n) => s.split(/\\s+/).slice(0, n).join(' ').trim();
                const searchParts = [text];
                if (text.length > 20) searchParts.push(firstWords(text, 5), firstWords(text, 3));
                if (/\\s/.test(text)) searchParts.push(collapseSpaces(text));
                const vh = window.innerHeight || 720;
                const getParentLabel = (el) => {
                    let p = el;
                    for (let i = 0; i < 8 && p; i++) {
                        p = p.parentElement;
                        if (!p) break;
                        const h = p.querySelector ? p.querySelector('h1, h2, h3, h4, [class*="title"], [class*="name"]') : null;
                        if (h && h.textContent) {
                            const t = (h.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                            if (t.length > 2 && t.length < 150) return t;
                        }
                        if (p.getAttribute && (p.getAttribute('data-testid') || (p.className && /card|item|block/.test(String(p.className)))) && p.textContent) {
                            const first = (p.textContent || '').trim().split(/\\n|\\r/)[0].replace(/\\s+/g, ' ').slice(0, 60);
                            if (first.length > 2) return first;
                        }
                    }
                    return null;
                };
                const clickableSel = 'a, button, [role="button"], input[type="submit"], input[type="button"]';
                const getText = (el) => norm(el.textContent || el.innerText || el.value || el.getAttribute('aria-label') || '');
                const tryClick = (el) => {
                    el.scrollIntoView({ behavior: 'instant', block: 'center' });
                    const r = el.getBoundingClientRect();
                    const x = r.left + r.width / 2, y = r.top + r.height / 2;
                    const o = { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y };
                    ['mouseenter', 'mouseover', 'mousedown', 'mouseup', 'click'].forEach(ev =>
                        el.dispatchEvent(new MouseEvent(ev, { ...o, button: 0 }))
                    );
                    if (el.click) el.click();
                };
                const collectAllMatchesWithEl = (txt) => {
                    const arr = [];
                    const requestWordCount = txt.split(/\\s+/).filter(w => w.length > 0).length;
                    const txtCol = collapseSpaces(txt);
                    for (const el of document.querySelectorAll(clickableSel)) {
                        const c = getText(el);
                        if (!c) continue;
                        const elementWordCount = c.split(/\\s+/).filter(w => w.length > 0).length;
                        const elementContainsRequest = c.includes(txt);
                        const exactMatch = c === txt;
                        const requestContainsElement = txt.includes(c) && !(requestWordCount >= 3 && elementWordCount <= 2);
                        const cCol = collapseSpaces(c);
                        const collapsedMatch = txtCol.length >= 2 && (cCol === txtCol || cCol.includes(txtCol) || (cCol.length >= 4 && txtCol.includes(cCol)));
                        if (elementContainsRequest || exactMatch || requestContainsElement || collapsedMatch) {
                            const ctx = getParentLabel(el);
                            const rect = el.getBoundingClientRect();
                            arr.push({ el, i: arr.length, text: (el.textContent || '').trim().slice(0, 50), ctx: ctx || '', inView: rect.top < vh && rect.bottom > 0 });
                        }
                    }
                    return arr;
                };
                const findAndClick = (txt) => {
                    const txtCol = collapseSpaces(txt);
                    const allMatches = collectAllMatchesWithEl(txt);
                    const requestWordCount = txt.split(/\\s+/).filter(w => w.length > 0).length;
                    if (allMatches.length === 0) {
                        const candidates = [];
                        for (const el of document.querySelectorAll('*')) {
                            const c = norm(el.textContent || el.innerText || el.getAttribute('aria-label') || '');
                            if (!c || c.length >= 400) continue;
                            const elementWordCount = c.split(/\\s+/).filter(w => w.length > 0).length;
                            const elementContainsRequest = c.includes(txt);
                            const elementStartsWith = c.startsWith(txt);
                            const requestContainsElement = txt.includes(c.slice(0, 60)) && !(requestWordCount >= 3 && elementWordCount <= 2);
                            const cColFb = collapseSpaces(c);
                            const collapsedMatch = txtCol.length >= 2 && (cColFb === txtCol || cColFb.includes(txtCol) || (cColFb.length >= 4 && txtCol.includes(cColFb)));
                            if (elementContainsRequest || elementStartsWith || requestContainsElement || collapsedMatch) {
                                const clickables = el.querySelectorAll ? Array.from(el.querySelectorAll(clickableSel)) : [];
                                let tgt = null;
                                let fallbackChoseByText = false;
                                if (clickables.length > 0) {
                                    const byExactMatch = clickables.find(cc => { const ct = getText(cc); return ct === txt; });
                                    if (byExactMatch) {
                                        tgt = byExactMatch;
                                        fallbackChoseByText = true;
                                    } else {
                                        const matching = clickables.filter(cc => { const ct = getText(cc); return ct && (ct.includes(txt) || txt.includes(ct)); });
                                        if (matching.length > 0) {
                                            const best = matching.filter(cc => { const ct = getText(cc); return ct.includes(txt) || ct === txt; });
                                            tgt = (best.length > 0 ? best : matching)[0];
                                            fallbackChoseByText = true;
                                        } else {
                                            const withText = clickables.filter(cc => (getText(cc) || '').trim().length >= 2);
                                            tgt = withText.length > 0 ? withText[0] : clickables[0];
                                        }
                                    }
                                } else {
                                    tgt = el.querySelector ? el.querySelector(clickableSel) : null;
                                }
                                if (!tgt) tgt = el;
                                if (tgt && tgt.getBoundingClientRect) {
                                    const rect = tgt.getBoundingClientRect();
                                    const inView = rect.top < vh && rect.bottom > 0;
                                    candidates.push({ el: tgt, textLen: c.length, inView: inView });
                                }
                            }
                        }
                        candidates.sort((a, b) => {
                            if (a.inView !== b.inView) return a.inView ? -1 : 1;
                            return a.textLen - b.textLen;
                        });
                        if (candidates.length > 0) {
                            const best = candidates[0];
                            const tgt = best.el;
                            tryClick(tgt);
                            return { ok: true, matched: (tgt.textContent || '').trim().slice(0, 80), clickedContext: getParentLabel(tgt), allMatchesCount: 0, fallbackChoseByText: true };
                        }
                        return null;
                    }
                    let toClick = allMatches[0];
                    if (allMatches.length > 1) {
                        const best = allMatches.filter(m => { const t = getText(m.el); return t.includes(txt) || t === txt; });
                        toClick = (best.length > 0 ? best : allMatches)[0];
                    }
                    const bestForSpecific = allMatches.filter(m => { const t = getText(m.el); return t.includes(txt) || t === txt; });
                    const hasUniqueSpecificMatch = bestForSpecific.length === 1;
                    if (allMatches.length >= 3) {
                        const textIsShort = raw.split(/\\s+/).filter(w => w.length > 0).length <= 2;
                        const cannotDisambiguate = !hasUniqueSpecificMatch;
                        if (textIsShort && cannotDisambiguate) {
                            return {
                                ok: false,
                                ambiguous: true,
                                requestedText: raw.slice(0, 80),
                                count: allMatches.length,
                                contexts: allMatches.slice(0, 5).map(m => ({
                                    text: m.text.slice(0, 40),
                                    context: (m.ctx || '').slice(0, 60)
                                }))
                            };
                        }
                    }
                    tryClick(toClick.el);
                    return {
                        ok: true,
                        matched: (toClick.el.textContent || '').trim().slice(0, 80),
                        clickedContext: toClick.ctx || getParentLabel(toClick.el),
                        allMatchesCount: allMatches.length,
                        allMatchesPreview: allMatches.slice(0, 15).map(m => ({ i: m.i, text: m.text.slice(0, 40), ctx: (m.ctx || '').slice(0, 50), inView: m.inView })),
                        clickedIndex: allMatches.indexOf(toClick)
                    };
                };
                for (const t of searchParts) {
                    if (t.length < 2) continue;
                    const res = findAndClick(t);
                    if (res) return res;
                }
                const sample = Array.from(document.querySelectorAll(clickableSel)).slice(0, 5)
                    .map(el => ({ t: getText(el).slice(0, 80), tag: el.tagName }));
                return { ok: false, debug: { searchParts, sample } };
            }
        """

        _JS_CLICK_IN_DIALOG_SCRIPT = """
            ([textArg]) => {
                const raw = (textArg || '').trim();
                if (!raw) return { ok: false };
                const root = document.querySelector('[role="dialog"]') || document.querySelector('[aria-modal="true"]');
                if (!root) return { ok: false };
                const wsp = new RegExp('[\\\\s\\\\u00A0\\\\u2009\\\\u202F]+', 'g');
                const norm = (s) => (s || '').replace(wsp, ' ').trim().toLowerCase();
                const collapseSpaces = (s) => (s || '').replace(/\\\\s+/g, '').toLowerCase();
                const text = norm(raw);
                const textCol = collapseSpaces(raw);
                const getText = (el) => {
                    let t = norm(el.textContent || el.innerText || el.value || el.getAttribute('aria-label') || '');
                    if (!t && el.type && (el.type === 'radio' || el.type === 'checkbox')) {
                        const lab = el.closest ? el.closest('label') : null;
                        const lab2 = el.id ? document.querySelector('label[for=\"' + el.id + '\"]') : null;
                        if (lab) t = norm(lab.textContent || lab.innerText || '');
                        else if (lab2) t = norm(lab2.textContent || lab2.innerText || '');
                    }
                    return t;
                };
                const modalContext = (root.innerText || root.textContent || '').trim().replace(wsp, ' ').split('\\n')[0].slice(0, 80);
                const isDisabled = (el) => {
                    if (el.disabled || el.getAttribute('aria-disabled') === 'true') return true;
                    const style = window.getComputedStyle(el);
                    if (style.pointerEvents === 'none' || style.display === 'none' || style.visibility === 'hidden') return true;
                    return false;
                };
                const tryClick = (el) => {
                    el.scrollIntoView({ behavior: 'instant', block: 'center' });
                    const r = el.getBoundingClientRect();
                    const x = r.left + r.width / 2, y = r.top + r.height / 2;
                    const o = { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y };
                    ['mouseenter', 'mouseover', 'mousedown', 'mouseup', 'click'].forEach(ev =>
                        el.dispatchEvent(new MouseEvent(ev, { ...o, button: 0 }))
                    );
                    if (el.click) el.click();
                };
                const clickableSel = 'a, button, [role="button"], [role="option"], [role="menuitemradio"], [role="menuitemcheckbox"], input[type="submit"], input[type="radio"], input[type="checkbox"]';
                const btns = Array.from(root.querySelectorAll(clickableSel));
                for (const el of btns) {
                    const c = getText(el);
                    const cCol = collapseSpaces(c);
                    const collapsedOk = textCol.length >= 2 && (cCol === textCol || cCol.includes(textCol) || (cCol.length >= 4 && textCol.includes(cCol)));
                    const match = c && (c.includes(text) || c === text || text.includes(c) || collapsedOk);
                    if (match) {
                        if (isDisabled(el)) {
                            return { ok: false, disabled: true, btnText: c, reason: 'Кнопка недоступна. Сначала выбери опцию в модалке (например, чекбокс, переключатель или вариант из списка).' };
                        }
                        tryClick(el);
                        return { ok: true, matched: (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0, 60), modalContext };
                    }
                }
                const all = Array.from(root.querySelectorAll('*'));
                const candidates = [];
                for (const el of all) {
                    const c = getText(el);
                    const cCol = collapseSpaces(c);
                    const collapsedOkAll = textCol.length >= 2 && (cCol === textCol || cCol.includes(textCol) || (cCol.length >= 4 && textCol.includes(cCol)));
                    const matchAll = c && c.length < 200 && (c.includes(text) || c.startsWith(text) || collapsedOkAll);
                    if (matchAll) {
                        const clickables = el.querySelectorAll ? Array.from(el.querySelectorAll(clickableSel)) : [];
                        let target = null;
                        if (clickables.length > 1) {
                            const byText = clickables.find(cc => { const ct = getText(cc); return ct && (ct === text || ct.includes(text) || text.includes(ct)); });
                            target = byText || clickables[0];
                        } else {
                            target = clickables[0] || el.querySelector ? el.querySelector(clickableSel) : null;
                        }
                        if (!target) target = el;
                        if (target && target.getBoundingClientRect)
                            candidates.push({ el: target, len: c.length });
                    }
                }
                if (candidates.length > 0) {
                    const fullMatch = candidates.filter(x => {
                        const ct = getText(x.el);
                        return ct && (ct.includes(text) || ct === text);
                    });
                    const toUse = fullMatch.length > 0 ? fullMatch : candidates;
                    toUse.sort((a, b) => a.len - b.len);
                    const best = toUse[0].el;
                    const mt = (getText(best) || (best.textContent || '').trim()).slice(0, 60);
                    return { ok: true, matched: mt, modalContext, usePlaywrightClick: true };
                }
                return { ok: false };
            }
        """

        async def _do_click(force_click: bool = False) -> dict[str, Any]:
            s = effective_scope if use_text else scope
            if use_sel:
                loc = s.locator(use_sel).first
                url_before_click = self._page.url
                await loc.click(timeout=ACTION_TIMEOUT_MS, no_wait_after=True, force=force_click)
            elif use_text:
                url_before_click = self._page.url
                if s != self._page:
                    if DIAG and use_text:
                        try:
                            diag_d = await self._page.evaluate(_CHECK_DIALOG_SCRIPT)
                            print("[DIAG] dialog_diag (scope=dialog) hasDialog=%s btnCount=%s btnTexts=%s primaryBtn=%s" % (
                                diag_d.get("hasDialog"),
                                diag_d.get("btnCount"),
                                json.dumps(diag_d.get("btnTexts", []), ensure_ascii=False)[:300],
                                (diag_d.get("primaryActionBtn") or "")[:40],
                            ), file=sys.stderr)
                        except Exception as ed:
                            print("[DIAG] dialog_diag error: %s" % str(ed)[:80], file=sys.stderr)
                    result_d = await self._page.evaluate(_JS_CLICK_IN_DIALOG_SCRIPT, [use_text])
                    if not (isinstance(result_d, dict) and result_d.get("ok")):
                        if isinstance(result_d, dict) and result_d.get("disabled"):
                            btn_text = result_d.get("btnText", "")
                            reason = result_d.get("reason", "Кнопка недоступна")
                            if DIAG:
                                print("[DIAG] click_element dialog DISABLED btn=%r reason=%r" % (btn_text[:40], reason[:80]), file=sys.stderr)
                            return {"success": False, "error": reason, "suggestion": "Выбери опцию в модалке (чекбокс/переключатель/селект) и снова нажми кнопку."}
                        if DIAG:
                            try:
                                dd = await self._page.evaluate(_CHECK_DIALOG_SCRIPT)
                                print("[DIAG] click_element dialog FAILED search=%r dialog_btnTexts=%s" % (
                                    (use_text or "")[:50],
                                    json.dumps(dd.get("btnTexts", []), ensure_ascii=False)[:300],
                                ), file=sys.stderr)
                            except Exception:
                                print("[DIAG] click_element dialog: element not found search=%r" % (use_text or "")[:50], file=sys.stderr)
                        txt_lower = (use_text or "").strip().lower()
                        if re.search(r"закрыть\s*модальн|close\s*modal", txt_lower):
                            try:
                                await self._page.keyboard.press("Escape")
                                await asyncio.sleep(0.3)
                                if DIAG:
                                    print("[DIAG] click_element: close-modal fallback -> Escape", file=sys.stderr)
                            except Exception:
                                pass
                            return {"success": True}
                        return {"success": False, "error": "Элемент не найден в диалоге", "suggestion": "Проверь текст кнопок в модальном окне."}
                    if result_d.get("usePlaywrightClick") and result_d.get("matched"):
                        try:
                            await s.get_by_text(result_d["matched"], exact=False).first.click(timeout=ACTION_TIMEOUT_MS, no_wait_after=True)
                        except Exception as pe:
                            if DIAG:
                                print("[DIAG] click_element dialog Playwright fallback err: %s" % str(pe)[:80], file=sys.stderr)
                            return {
                                "success": False,
                                "error": "Элемент в диалоге не кликается: %s" % str(pe)[:80],
                                "suggestion": "Проверь текст кнопок в модальном окне.",
                            }
                    elif result_d.get("usePlaywrightClick"):
                        try:
                            await s.get_by_text(use_text or "", exact=False).first.click(timeout=ACTION_TIMEOUT_MS, no_wait_after=True)
                        except Exception as pe:
                            if DIAG:
                                print("[DIAG] click_element dialog Playwright fallback err: %s" % str(pe)[:80], file=sys.stderr)
                            return {
                                "success": False,
                                "error": "Элемент в диалоге не кликается: %s" % str(pe)[:80],
                                "suggestion": "Проверь текст кнопок в модальном окне.",
                            }
                else:
                    result_js = await self._page.evaluate(_JS_CLICK_SCRIPT, [use_text])
                    if not (isinstance(result_js, dict) and result_js.get("ok")):
                        if isinstance(result_js, dict) and result_js.get("ambiguous"):
                            count = result_js.get("count", 0)
                            req_text = result_js.get("requestedText", use_text or "")
                            contexts = result_js.get("contexts", [])
                            ctx_preview = "; ".join([f"«{c.get('context', '')}»" for c in contexts[:3] if c.get('context')])
                            if not ctx_preview:
                                ctx_preview = "; ".join([f"текст «{c.get('text', '')}»" for c in contexts[:3]])
                            if DIAG:
                                print("[DIAG] click_element AMBIGUOUS request=%r count=%d contexts=%s" % (
                                    req_text[:50], count, json.dumps(contexts, ensure_ascii=False)[:400]
                                ), file=sys.stderr)
                            return {
                                "success": False,
                                "error": f"Неоднозначный запрос: найдено {count} элементов с текстом «{req_text}».",
                                "suggestion": f"Уточни запрос, указав контекст рядом с элементом. Примеры найденных: {ctx_preview}. Или кликни сначала нужный элемент (блок/контейнер), затем кнопку в открывшемся окне."
                            }
                        if DIAG and isinstance(result_js, dict) and result_js.get("debug"):
                            d = result_js["debug"]
                            print("[DIAG] click_element JS debug searchParts=%r sample=%s" % (
                                d.get("searchParts", []),
                                json.dumps(d.get("sample", []), ensure_ascii=False)[:300],
                            ), file=sys.stderr)
                        word_count = len((use_text or "").split())
                        if word_count > 10:
                            first_words = " ".join((use_text or "").split()[:3])
                            return {
                                "success": False,
                                "error": f"Элемент не найден. Запрос слишком длинный ({word_count} слов).",
                                "suggestion": f"Используй первые слова запроса, например: «{first_words}»."
                            }
                        return {"success": False, "error": "Элемент не найден", "suggestion": "Проверь текст по get_page_content; для длинного запроса — первые слова."}
                    if DIAG:
                        m = result_js.get("matched", "")
                        ctx = result_js.get("clickedContext", "")
                        n = result_js.get("allMatchesCount")
                        fallback_by_text = result_js.get("fallbackChoseByText", False)
                        pre = result_js.get("allMatchesPreview", [])
                        pre_s = json.dumps(pre[:10], ensure_ascii=False)[:500] if pre else ""
                        if fallback_by_text:
                            print("[DIAG] click_element fallback chose by text (avoided first-descendant) matched=%r clickedContext=%r" % (m[:50] if m else "", ctx[:60] if ctx else ""), file=sys.stderr)
                        elif not fallback_by_text:
                            print("[DIAG] click_element page done matched=%r clickedContext=%r allMatches=%s preview=%s" % (m[:50] if m else "", ctx[:60] if ctx else "", n, pre_s), file=sys.stderr)
            else:
                return {"success": False, "error": "Укажи text или selector"}
            self._last_click_time = time.monotonic()
            await asyncio.sleep(0.8)
            url_after = self._page.url
            page_navigated = url_after != url_before_click
            result = {"success": True}
            if page_navigated:
                result["page_navigated"] = True
            if force_click and (use_sel or use_text):
                result["force_used"] = True
            return result

        try:
            return await _do_click(force_click=False)
        except Exception as e:
            err = str(e)
            err_lower = err.lower()
            if ("timeout" in err_lower or "exceeded" in err_lower) and (use_sel or use_text):
                if DIAG:
                    print("[DIAG] click_element timeout, retrying with force=True", file=sys.stderr)
                try:
                    r = await _do_click(force_click=True)
                    print("  [click] force=True succeeded", file=sys.stderr)
                    return r
                except Exception:
                    pass
            sug = "Попробуй другой способ (другой текст/селектор) или scroll перед кликом."
            if scope != self._page and ("timeout" in err_lower or "exceeded" in err_lower):
                sug = "Диалог открыт: ищи элементы только внутри него."
            if DIAG:
                print("[DIAG] click_element EXCEPTION err=%s text=%r sel=%r scope=%s" % (err[:150], (text or "")[:40], (sel or "")[:40], "dialog" if scope != self._page else "page"), file=sys.stderr)
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
        _CHECK_DIALOG_INPUT_SCRIPT = """
            ([placeholderArg, fieldIndexArg]) => {
                const d = document.querySelector('[role="dialog"]') || document.querySelector('[aria-modal="true"]');
                if (!d) return { hasDialog: false };
                const inputSel = 'input:not([type=checkbox]):not([type=radio]):not([type=hidden]), textarea, [contenteditable="true"]';
                const inputs = Array.from(d.querySelectorAll(inputSel)).filter(el => {
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' && style.visibility !== 'hidden' && (el.offsetParent !== null || style.position === 'fixed');
                });
                const placeholders = inputs.map(el => (el.getAttribute('placeholder') || el.getAttribute('name') || el.getAttribute('aria-label') || '').trim()).filter(Boolean);
                const placeholderNorm = (placeholderArg || '').trim().toLowerCase();
                let hasMatchingInput = false;
                if (placeholderNorm) {
                    for (let i = 0; i < inputs.length; i++) {
                        const p = (inputs[i].getAttribute('placeholder') || inputs[i].getAttribute('name') || inputs[i].getAttribute('aria-label') || '').trim().toLowerCase();
                        if (p && (p.indexOf(placeholderNorm) >= 0 || placeholderNorm.indexOf(p) >= 0)) {
                            hasMatchingInput = true;
                            break;
                        }
                    }
                }
                const idx = fieldIndexArg != null ? parseInt(fieldIndexArg, 10) : null;
                const fieldIndexInDialog = (idx != null && !isNaN(idx) && idx >= 1) ? (idx <= inputs.length) : true;
                return {
                    hasDialog: true,
                    inputCountInDialog: inputs.length,
                    placeholdersInDialog: placeholders.slice(0, 20),
                    hasMatchingPlaceholder: placeholderNorm ? hasMatchingInput : null,
                    fieldIndexInDialog: idx == null ? null : fieldIndexInDialog
                };
            }
        """
        effective_scope = scope
        if scope != self._page and (placeholder or field_index_raw is not None):
            try:
                diag_in = await self._page.evaluate(
                    _CHECK_DIALOG_INPUT_SCRIPT,
                    [placeholder or "", field_index_raw],
                )
                if isinstance(diag_in, dict) and diag_in.get("hasDialog"):
                    use_page = False
                    if placeholder and diag_in.get("hasMatchingPlaceholder") is False:
                        use_page = True
                        if DIAG:
                            print(
                                "[DIAG] type_text: placeholder %r not in dialog, using page scope; placeholdersInDialog=%s"
                                % (
                                    (placeholder or "")[:50],
                                    json.dumps(diag_in.get("placeholdersInDialog", [])[:15], ensure_ascii=False)[:300],
                                ),
                                file=sys.stderr,
                            )
                    if field_index_raw is not None and diag_in.get("fieldIndexInDialog") is False:
                        use_page = True
                        if DIAG:
                            print(
                                "[DIAG] type_text: field_index=%s > inputCountInDialog=%s, using page scope"
                                % (field_index_raw, diag_in.get("inputCountInDialog", 0)),
                                file=sys.stderr,
                            )
                    if use_page:
                        effective_scope = self._page
            except Exception:
                pass
        if DIAG:
            print("[DIAG] type_text effective_scope=%s placeholder=%r field_index=%s selector=%r" % (
                "page" if effective_scope == self._page else "dialog",
                (placeholder or "")[:40],
                field_index_raw,
                (sel or "")[:50] if sel else None,
            ), file=sys.stderr)
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
                await effective_scope.locator(sel).first.fill(text, timeout=ACTION_TIMEOUT_MS)
            elif placeholder:
                await effective_scope.get_by_placeholder(placeholder).first.fill(text, timeout=ACTION_TIMEOUT_MS)
            elif field_index is not None:
                loc = effective_scope.locator(combined_selector).nth(field_index - 1)
                await loc.fill(text, timeout=ACTION_TIMEOUT_MS)
            elif len(text) > 80:
                txt = effective_scope.locator("textarea:visible")
                if await txt.count() > 0:
                    await txt.first.fill(text, timeout=ACTION_TIMEOUT_MS)
                else:
                    loc = effective_scope.locator(
                        f"{input_ok}, [contenteditable='true']"
                    ).first
                    await loc.fill(text, timeout=ACTION_TIMEOUT_MS)
            else:
                loc = effective_scope.locator(combined_selector).first
                await loc.fill(text, timeout=ACTION_TIMEOUT_MS)
            await asyncio.sleep(0.3)
            return {"success": True}
        except Exception as e:
            err = str(e)
            err_lower = err.lower()
            suf = " Вызови get_page_content и укажи placeholder, selector или field_index (1, 2, 3…) по порядку полей; для длинного текста подойдёт textarea."
            if ("timeout" in err_lower or "exceeded" in err_lower) and (
                "input" in err_lower or "textarea" in err_lower or "placeholder" in err_lower or "locator(" in err_lower
            ):
                suf += " Если форма с полями ещё не открыта — сначала открой её (кнопка/ссылка на странице), затем get_page_content и заполняй поля по placeholder/selector/field_index."
            if effective_scope != self._page:
                suf += " Диалог открыт: используй placeholder или selector для поля внутри него."
            if field_index is not None:
                suf += " Если field_index — проверь, что номер не больше числа полей (в списке inputs не считай checkbox/radio)."
            suf += " Нумерация field_index — из строки «Поля для ввода» в начале get_page_content."
            if DIAG:
                print("[DIAG] type_text EXCEPTION err=%s placeholder=%r field_index=%s sel=%r scope=%s" % (err[:120], (placeholder or "")[:30], field_index_raw, (sel or "")[:30], "dialog" if effective_scope != self._page else "page"), file=sys.stderr)
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
                if DIAG:
                    print(
                        "[DIAG] scroll container=%r direction=%s amount=%d scrollTop=%s scrollH=%s clientH=%s"
                        % (
                            container_sel[:60],
                            direction,
                            amount,
                            result.get("scrollTop"),
                            result.get("scrollHeight"),
                            result.get("clientHeight"),
                        ),
                        file=sys.stderr,
                    )
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            try:
                result = await self._page.evaluate(
                    """
                    (delta) => {
                        const d = document.querySelector('[role="dialog"]') || document.querySelector('[role="alertdialog"]') || document.querySelector('[aria-modal="true"]');
                        if (d && d.getBoundingClientRect().width > 0 && d.getBoundingClientRect().height > 0) {
                            const before = d.scrollTop || 0;
                            d.scrollTop = Math.max(0, before + delta);
                            return { target: 'dialog', scrollTop: d.scrollTop, scrollHeight: d.scrollHeight, clientHeight: d.clientHeight };
                        }
                        window.scrollBy(0, delta);
                        return { target: 'window', scrollY: window.scrollY };
                    }
                    """,
                    delta,
                )
                if DIAG and isinstance(result, dict):
                    if result.get("target") == "dialog":
                        print(
                            "[DIAG] scroll target=dialog direction=%s amount=%d scrollTop=%s scrollH=%s clientH=%s"
                            % (direction, amount, result.get("scrollTop"), result.get("scrollHeight"), result.get("clientHeight")),
                            file=sys.stderr,
                        )
                    else:
                        print(
                            "[DIAG] scroll target=window direction=%s amount=%d scrollY=%s"
                            % (direction, amount, result.get("scrollY")),
                            file=sys.stderr,
                        )
            except Exception as e:
                if DIAG:
                    print("[DIAG] scroll evaluate error: %s" % str(e)[:100], file=sys.stderr)
                await self._page.evaluate("d => window.scrollBy(0, d)", delta)

        await asyncio.sleep(0.4)
        out: dict[str, Any] = {"success": True}
        try:
            content = await self._fetch_page_content()
            out["content"] = content
            self._last_get_content_time = time.monotonic()
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
