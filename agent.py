#!/usr/bin/env python3
"""
Автономный AI‑агент для управления браузером через MCP.
Использует Hydra/OpenAI‑совместимый API, StdioServerParameters + stdio_client.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import anyio
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI

load_dotenv()

# Паттерны «опасных» действий — перед выполнением спрашиваем пользователя (security layer)
DANGEROUS_PATTERNS = re.compile(
    r"\b(оплатить|подтвердить\s*заказ|удалить\s*навсегда|оформить\s*заказ|"
    r"pay|checkout|confirm\s*order|delete\s*permanently|"
    r"подтвердить\s*оплату|купить|buy)\b",
    re.IGNORECASE,
)

# ANSI (только если вывод в TTY)
_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _dim(s: str) -> str:
    return "\033[90m%s\033[0m" % s if _TTY else s


def _em(s: str) -> str:
    return "\033[1m%s\033[0m" % s if _TTY else s


def _green(s: str) -> str:
    return "\033[92m%s\033[0m" % s if _TTY else s


def _yellow(s: str) -> str:
    return "\033[93m%s\033[0m" % s if _TTY else s


def _sep(w: int = 52) -> str:
    return "─" * w


def _openai_client() -> OpenAI:
    base = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY не задан. Заполни .env (см. env.example).")
    if base:
        return OpenAI(api_key=key, base_url=base)
    return OpenAI(api_key=key)


def _model() -> str:
    return (os.getenv("OPENAI_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514").strip()


def _agent_dir() -> Path:
    return Path(__file__).resolve().parent


def _mcp_tool_to_openai(t) -> dict:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.inputSchema if hasattr(t, "inputSchema") and t.inputSchema else {"type": "object"},
        },
    }


async def _read_line(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return (await loop.run_in_executor(None, lambda: input(prompt))).strip()


async def _confirm(prompt: str) -> bool:
    line = await _read_line(prompt)
    return line.lower() in ("y", "yes", "д", "да")


def _parse_tool_result(content: list, structured: dict | None, is_error: bool) -> dict:
    if is_error:
        for blk in content:
            if getattr(blk, "type", None) == "text" and getattr(blk, "text", None):
                try:
                    return json.loads(blk.text)
                except json.JSONDecodeError:
                    return {"success": False, "error": blk.text}
        return {"success": False, "error": "Unknown tool error"}
    if structured is not None:
        return structured
    for blk in content:
        if getattr(blk, "type", None) == "text" and getattr(blk, "text", None):
            try:
                return json.loads(blk.text)
            except json.JSONDecodeError:
                return {"result": blk.text}
    return {"success": True}


def _msg_content_str(msg) -> str:
    """Безопасно извлечь текстовый content из сообщения (str или list of parts)."""
    c = getattr(msg, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(parts).strip()
    return ""


def _fmt_mins(seconds: float) -> str:
    if seconds < 60:
        return "%.1f с" % seconds
    m = int(seconds // 60)
    s = seconds % 60
    if s < 0.1:
        return "%d мин" % m
    return "%d мин %.1f с" % (m, s)


def _format_tool_result(name: str, payload: dict) -> str:
    """Краткая строка для вывода результата инструмента (без JSON-мусора)."""
    err = payload.get("error") or (payload.get("message") if not payload.get("success", True) else None)
    if err:
        s = str(err)
        return "✗ %s" % (s[:120] + "…" if len(s) > 120 else s)
    if name in ("navigate", "go_back"):
        url = payload.get("url") or ""
        title = (payload.get("title") or "").strip()
        if title:
            t = title[:50]
            return "✓ %s" % (t + "…" if len(title) > 50 else t)
        return "✓ %s" % (url[:60] + "…" if len(url) > 60 else url)
    if name == "get_page_content":
        c = payload.get("content") or {}
        text = c.get("text") if isinstance(c, dict) else ""
        n = len(text) if isinstance(text, str) else 0
        suf = " + диалог" if (isinstance(c, dict) and c.get("modal")) else ""
        return "✓ страница, ~%d символов%s" % (n, suf)
    if name == "finish_task":
        return "✓ итог"
    if name == "wait_for_user":
        return "✓ продолжено"
    return "✓"


# Синтетические инструменты (обрабатываются в агенте, не в MCP)
WAIT_FOR_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "wait_for_user",
        "description": "Ожидание ввода от пользователя в браузере. Используется для форм входа/регистрации (пароли не проходят через LLM) и для капчи. Пользователь заполняет форму или решает капчу вручную в браузере, затем вводит «готово» или «done». После возврата сделай get_page_content и продолжай задачу.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

FINISH_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_task",
        "description": "Вызови, когда задача полностью выполнена. Обязательно передай summary — краткий итог (что сделано, результат). После вызова агент завершает выполнение и выводит итог пользователю.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Краткий итог: что сделано, какой результат.",
                },
                "success": {
                    "type": "boolean",
                    "description": "Задача выполнена успешно?",
                    "default": True,
                },
            },
            "required": ["summary"],
        },
    },
}

READY_TRIGGERS = frozenset({"готово", "done", "ok", "продолжай", "go", "yes", "да"})


async def _do_wait_for_user() -> dict:
    print()
    print(_yellow("  ⏸ Ожидание: войдите в аккаунт или решите капчу в браузере."))
    print(_dim("     Напишите «готово» или «done» и нажмите Enter, когда закончите."))
    print()
    while True:
        line = (await _read_line("  Готово? (готово/done) > ")).lower().strip()
        if line in READY_TRIGGERS:
            return {"success": True, "message": "Пользователь готов. Продолжаю."}
        print(_dim("     Введите «готово» или «done», чтобы продолжить."))


async def run_agent() -> None:
    client = _openai_client()
    model = _model()
    cwd = str(_agent_dir())
    env = {**os.environ}

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        cwd=cwd,
        env=env,
    )

    system = """Ты автономный AI‑агент, управляющий браузером через MCP‑инструменты.

Правила:
1. Сначала анализируй страницу (get_page_content), потом действуй (click_element, type_text и т.д.).
2. Ищи элементы по видимому тексту или по типу; не придумывай селекторы — используй то, что видишь на странице.
2a. Одна вкладка: новые вкладки запрещены, все ссылки открываются в текущей. Кликай по ссылкам по одной; после клика — get_page_content. Чтобы вернуться назад, используй go_back.
3. При ошибке инструмента пробуй другой способ (другой текст, scroll, другой элемент).
3a. Если открыто модальное окно (в get_page_content есть [Модальное окно] или modal): работай только с ним. Клики и ввод — только внутри. Передавай text или selector элемента, не [role=dialog]. Для ввода — placeholder или selector поля.
4. Работай автономно, пока задача не выполнена или не понадобится уточнение у пользователя.
5. Перед деструктивными действиями при клике по «опасному» тексту система сама спросит подтверждение у пользователя — отдельный инструмент вызывать не нужно.
6. Перед каждым вызовом инструментов пиши 1–2 короткие фразы: что делаешь и зачем. Это показывается пользователю в логе и помогает понимать ход выполнения.
7. Сессии сохраняются автоматически при выходе. Если пользователь вручную вошёл в аккаунт в браузере (логин/пароль), вызови save_session — тогда при следующих запусках вводить логин/пароль снова не придётся.
8. Когда задача полностью выполнена — вызови finish_task с summary (краткий итог). Не завершай задачу длинным текстом вместо вызова finish_task.

Чувствительные действия: для форм входа и капчи — wait_for_user (не вводи пароли, не кликай по кнопке входа). Для платежей, удалений и т.п. система сама запросит подтверждение.

Логин и капча:
• При формах входа (логин/пароль) или капче вызывай wait_for_user. Пользователь заполнит форму или решит капчу в браузере и напишет «готово»/«done». После возврата — get_page_content и продолжай задачу.

Завершение: когда всё сделано — вызови finish_task с summary (итог и результат). Итог затем показывается пользователю."""

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tools = list(tools_result.tools)
            openai_tools = [_mcp_tool_to_openai(t) for t in tools] + [WAIT_FOR_USER_TOOL, FINISH_TASK_TOOL]

            print()
            print(_sep())
            print(_em("  Browser Agent (MCP)"))
            print(_sep())
            print(_dim("  Модель: %s") % model)
            print(_dim("  Введи задачу и нажми Enter. Выход: quit / exit / q / выход"))
            print(_sep())
            print()

            while True:
                task = await _read_line("\n  " + _em("Задача") + " > ")
                if not task or task.lower() in ("quit", "exit", "q", "выход"):
                    break

                t0 = time.monotonic()
                messages: list[dict] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": task},
                ]
                max_steps = 50
                done = False
                last_reply = ""
                step_used = 0
                empty_responses = 0
                failed_actions = 0

                print()
                print("  " + _sep(48))
                print(_green("  ▶ Начинаю выполнение"))
                print("  " + _sep(48))

                for step in range(max_steps):
                    step_used = step + 1
                    step_start = time.monotonic()
                    print()
                    print(_dim("  ── Шаг %d/%d ──") % (step + 1, max_steps))

                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=openai_tools,
                            tool_choice="auto",
                            temperature=0.1,
                            max_tokens=4096,
                        )
                    except Exception as e:
                        print(_yellow("  ✗ Ошибка LLM: %s") % e)
                        last_reply = "Ошибка LLM. Задача не выполнена."
                        break

                    msg = resp.choices[0].message
                    messages.append(msg.model_dump())

                    content_str = _msg_content_str(msg)
                    has_content = bool(content_str)
                    has_tools = bool(msg.tool_calls)
                    if not has_content and not has_tools:
                        empty_responses += 1
                        if empty_responses >= 2:
                            messages.append({
                                "role": "user",
                                "content": "Ответь действием (tool_calls) или вызови finish_task с итогом, если задача выполнена. Не отвечай пустым сообщением.",
                            })
                            empty_responses = 0
                            step_elapsed = time.monotonic() - step_start
                            print(_dim("  ⏱ %.1f с") % step_elapsed)
                            continue
                    else:
                        empty_responses = 0

                    def _brief(s: str, max_len: int = 180) -> str:
                        s = (s or "").strip()
                        if not s:
                            return ""
                        if len(s) <= max_len:
                            return s
                        u = s[:max_len].rsplit(maxsplit=1)
                        return (u[0] if u else s[:max_len]) + "…"

                    if content_str and msg.tool_calls:
                        brief = _brief(content_str)
                        if brief:
                            print(_dim("  💭 %s") % brief)

                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.function.name
                            try:
                                args = json.loads(tc.function.arguments or "{}")
                            except json.JSONDecodeError:
                                args = {}

                            if name == "click_element" and args.get("text"):
                                text = args.get("text") or ""
                                if DANGEROUS_PATTERNS.search(text):
                                    ok = await _confirm(
                                        _yellow('  Подтвердить действие "%s"? (y/n) ') % text
                                    )
                                    if not ok:
                                        result = {"success": False, "error": "Пользователь отклонил действие"}
                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tc.id,
                                            "content": json.dumps(result, ensure_ascii=False),
                                        })
                                        print(_dim("    ↳ пропущено по отказу"))
                                        continue

                            args_preview = json.dumps(args, ensure_ascii=False)
                            if len(args_preview) > 56:
                                args_preview = args_preview[:53] + "…"
                            print(_dim("    🛠 %s  %s") % (name, args_preview))

                            if name == "wait_for_user":
                                payload = await _do_wait_for_user()
                            elif name == "finish_task":
                                summary = (args.get("summary") or "").strip() or "Задача завершена."
                                success = args.get("success") if isinstance(args.get("success"), bool) else True
                                payload = {"success": success, "message": "Задача завершена.", "summary": summary}
                                done = True
                                last_reply = summary
                            else:
                                try:
                                    call_result = await session.call_tool(name, arguments=args)
                                    payload = _parse_tool_result(
                                        getattr(call_result, "content", []) or [],
                                        getattr(call_result, "structuredContent", None),
                                        getattr(call_result, "isError", False),
                                    )
                                except Exception as e:
                                    payload = {"success": False, "error": str(e)}
                            payload_str = json.dumps(payload, ensure_ascii=False)
                            short = _format_tool_result(name, payload)
                            print(_dim("       → %s") % short)
                            if not payload.get("success", True):
                                failed_actions += 1

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": payload_str,
                            })

                        step_elapsed = time.monotonic() - step_start
                        print(_dim("  ⏱ %.1f с") % step_elapsed)
                        await asyncio.sleep(0.3)
                        if done:
                            break
                        continue

                    if content_str:
                        last_reply = content_str
                        print()
                        for line in content_str.splitlines():
                            print("  " + line)
                        print()

                    step_elapsed = time.monotonic() - step_start
                    print(_dim("  ⏱ %.1f с") % step_elapsed)
                    await asyncio.sleep(0.2)

                elapsed = time.monotonic() - t0

                print()
                print("  " + _sep(48))
                print(_em("  ИТОГ"))
                print("  " + _sep(48))
                if last_reply:
                    for line in last_reply.splitlines():
                        print("  " + line)
                elif not done and step_used >= max_steps:
                    print(_yellow("  Достигнут лимит шагов (%d).") % max_steps)
                else:
                    print(_dim("  (краткий ответ выше)"))
                if failed_actions > 0:
                    print(_yellow("  Неудачных действий: %d (таймауты/ошибки кликов и т.п.)") % failed_actions)
                print()
                print(_green("  ⏱ Время: %s") % _fmt_mins(elapsed))
                print("  " + _sep(48))
                print()

    print()
    print(_dim("  Выход."))
    print()


def main() -> None:
    try:
        anyio.run(run_agent, backend="asyncio")
    except KeyboardInterrupt:
        print("\nПрервано (Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()
