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
# Таймаут клика/ввода (мс).
ACTION_TIMEOUT_MS = 8_000


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
            description="Получить содержимое текущей страницы: текст, кнопки, ссылки, поля ввода. Используй для анализа перед действиями.",
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
            description="Ввести текст в поле. Используй placeholder или selector, если полей несколько. Длинный текст вводится в textarea. Checkbox/radio не поддерживаются.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Текст для ввода"},
                    "selector": {"type": "string", "description": "CSS-селектор поля"},
                    "placeholder": {"type": "string", "description": "Placeholder поля для поиска"},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="scroll",
            description="Проскроллить страницу вверх или вниз.",
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"], "default": "down"},
                    "amount": {"type": "integer", "description": "Пиксели", "default": 500},
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
        }

    async def _get_page_content(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        include_html = bool(args.get("include_html"))
        try:
            content = await self._page.evaluate(
                """
                () => {
                    document.querySelectorAll('a[target="_blank"], a[target="_new"]').forEach(a => a.removeAttribute('target'));
                    const root = document.body;
                    const bodyText = root?.innerText ?? '';
                    const sel = (s, r) => (r || root).querySelectorAll(s);
                    const arr = (q) => Array.from(q);
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
                    const buttons = arr(sel('button, [role="button"], input[type="submit"]'))
                        .map(el => el.innerText?.trim() || el.value || el.getAttribute('aria-label') || '')
                        .filter(t => t).slice(0, 30);
                    const links = arr(sel('a[href]'))
                        .map(el => ({ text: (el.innerText || '').trim().slice(0, 200), href: el.href }))
                        .filter(l => l.text).slice(0, 50);
                    const inputs = arr(sel('input, textarea, select'))
                        .map(el => ({ type: el.type, placeholder: el.placeholder || '', name: el.name || '', id: el.id || '' }))
                        .slice(0, 30);
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
                    return {
                        text,
                        modal,
                        buttons,
                        links,
                        inputs,
                        url: window.location.href,
                        title: document.title
                    };
                }
                """
            )
            out: dict[str, Any] = {"success": True, "content": content}
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
        use_text = text
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
                await loc.click(timeout=ACTION_TIMEOUT_MS)
            elif use_text:
                loc = scope.get_by_text(use_text, exact=exact).first
                try:
                    await loc.click(timeout=ACTION_TIMEOUT_MS)
                except Exception:
                    if scope != self._page:
                        loc = scope.get_by_role("button", name=use_text).first
                        await loc.click(timeout=ACTION_TIMEOUT_MS)
                    else:
                        raise
            elif role:
                loc = scope.locator(f"[role='{role}']").first
                await loc.click(timeout=ACTION_TIMEOUT_MS)
            else:
                return {"success": False, "error": "Укажи text, selector или role"}
            await asyncio.sleep(0.8)
            return {"success": True}
        except Exception as e:
            err = str(e)
            sug = "Попробуй другой способ (другой текст/селектор) или scroll перед кликом."
            if scope != self._page and ("timeout" in err.lower() or "exceeded" in err.lower()):
                sug = "Диалог открыт: ищи элементы только внутри него. Элементы страницы под ним недоступны."
            return {"success": False, "error": err, "suggestion": sug}

    async def _type_text(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        text = args.get("text") or ""
        sel = args.get("selector")
        placeholder = args.get("placeholder")
        scope = await self._get_dialog_locator()
        if scope is None:
            scope = self._page
        else:
            try:
                await scope.scroll_into_view_if_needed(timeout=2_000)
            except Exception:
                pass
        input_ok = "input:not([type=checkbox]):not([type=radio]):visible"
        try:
            if sel:
                await scope.locator(sel).first.fill(text, timeout=ACTION_TIMEOUT_MS)
            elif placeholder:
                await scope.get_by_placeholder(placeholder).first.fill(text, timeout=ACTION_TIMEOUT_MS)
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
                loc = scope.locator(
                    f"{input_ok}, textarea:visible, [contenteditable='true']"
                ).first
                await loc.fill(text, timeout=ACTION_TIMEOUT_MS)
            await asyncio.sleep(0.3)
            return {"success": True}
        except Exception as e:
            err = str(e)
            suf = ""
            if scope != self._page:
                suf = " Диалог открыт: используй placeholder или selector для поля внутри него."
            return {"success": False, "error": err + suf}

    async def _scroll(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_browser()
        await self._ensure_single_tab()
        direction = (args.get("direction") or "down").lower()
        amount = max(0, int(args.get("amount") or 500))
        delta = amount if direction == "down" else -amount
        await self._page.evaluate("d => window.scrollBy(0, d)", delta)
        await asyncio.sleep(0.4)
        return {"success": True}

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
