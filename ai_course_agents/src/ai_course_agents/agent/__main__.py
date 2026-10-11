"""Консольный прогон агента — контрольная точка этапа 1 (без веба).

    uv run python -m ai_course_agents.agent "Сколько в Ростове круглосуточных аптек?"
    uv run python -m ai_course_agents.agent "Где кафе на Пушкинской?" "А какое из них ближе к Садовой, 50?"
    uv run python -m ai_course_agents.agent --no-thoughts "вопрос"   # не печатать размышления

Несколько вопросов задаются в ОДНОЙ сессии — так видно учёт контекста: второй вопрос
может ссылаться на первый («а какое из них…»).

Печатает события ADK по мере поступления — ровно то, что на этапе 3 уйдёт в браузер по SSE:
    💭 размышления модели (потоком)
    🔧 вызов инструмента с аргументами
    📦 результат инструмента (кратко)
    ✍️ ответ (потоком)
и в конце — шаги, токены, цена и время.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ai_course_agents.agent.agent import APP_NAME, build_agent
from ai_course_agents.agent.usage import Usage
from ai_course_agents.config import get_settings

DIM, RESET = "\033[2m", "\033[0m"     # серый цвет для размышлений (ANSI)


def short_result(response: dict) -> str:
    """Короткая сводка результата инструмента для консоли: не весь JSON, а суть."""
    if "error" in response:
        return f"ошибка: {response['error']}"
    if "places" in response:
        names = ", ".join(p["name"] for p in response["places"][:4])
        total = response.get("found_total")
        head = f"{len(response['places'])} мест" + (f" из {total}" if total is not None else "")
        return f"{head}: {names}{'…' if len(response['places']) > 4 else ''}"
    return json.dumps(response, ensure_ascii=False)[:200]


async def ask(runner: Runner, session_id: str, question: str, show_thoughts: bool) -> None:
    settings = get_settings()
    print(f"\n{'═' * 80}\n❓ {question}")
    usage, started, stream = Usage(), time.perf_counter(), None
    message = types.Content(role="user", parts=[types.Part(text=question)])
    # StreamingMode.SSE — текст и размышления приходят кусками (partial=True);
    # max_llm_calls — предохранитель от зацикливания (урок задачи 6 практики).
    config = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=settings.max_llm_calls)

    async for event in runner.run_async(user_id="console", session_id=session_id,
                                        new_message=message, run_config=config):
        usage.add(event)
        for part in (event.content.parts if event.content else []):
            if part.text and event.partial:
                kind = "thought" if part.thought else "text"
                if kind == "thought" and not show_thoughts:
                    continue
                if kind != stream:   # сменился поток — новая строка с подписью
                    print(f"\n{DIM}💭 " if kind == "thought" else f"{RESET}\n✍️  ", end="")
                    stream = kind
                print(part.text, end="", flush=True)
            elif part.function_call:
                args = json.dumps(part.function_call.args, ensure_ascii=False)
                print(f"{RESET}\n🔧 {part.function_call.name}({args})")
                stream = None
            elif part.function_response:
                print(f"📦 {short_result(part.function_response.response)}")
                stream = None

    print(RESET)
    u = usage.as_dict(settings)
    print(f"── запросов к LLM: {u['llm_calls']} · токены: вход {u['input_tokens']}, "
          f"(из кэша {u['cached_tokens']}), выход {u['output_tokens']} · цена ≈ {u['cost_rub']} ₽ · "
          f"{time.perf_counter() - started:.1f} с")


async def main(questions: list[str], show_thoughts: bool) -> None:
    settings = get_settings()
    if not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY пуст: впиши ключ в ai_course_agents/.env")
    sessions = InMemorySessionService()
    runner = Runner(app_name=APP_NAME, agent=build_agent(settings), session_service=sessions)
    session = await sessions.create_session(app_name=APP_NAME, user_id="console")
    for q in questions:
        await ask(runner, session.id, q, show_thoughts)


if __name__ == "__main__":
    args = sys.argv[1:]
    show = "--no-thoughts" not in args
    qs = [a for a in args if not a.startswith("--")] or ["Сколько в Ростове круглосуточных аптек?"]
    asyncio.run(main(qs, show))
