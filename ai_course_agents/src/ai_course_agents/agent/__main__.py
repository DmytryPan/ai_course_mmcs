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

Файл называется __main__.py — поэтому пакет можно запускать как `python -m ai_course_agents.agent`.
"""

from __future__ import annotations

import asyncio          # asyncio.run — запустить асинхронную main() из обычного кода
import json             # печать аргументов инструментов
import sys              # sys.argv — аргументы командной строки; sys.exit — выход с сообщением
import time             # time.perf_counter — замер времени ответа

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner                  # цикл агента ADK
from google.adk.sessions import InMemorySessionService  # история диалога в памяти процесса
from google.genai import types                         # типы сообщений: Content, Part

from ai_course_agents.agent.agent import APP_NAME, build_agent
from ai_course_agents.agent.usage import Usage
from ai_course_agents.config import get_settings

# Управляющие ANSI-последовательности терминала: \033[2m — тусклый (серый) текст,
# \033[0m — сбросить оформление. Размышления печатаем серым, чтобы отличать от ответа.
DIM, RESET = "\033[2m", "\033[0m"


def short_result(response: dict) -> str:
    """Короткая сводка результата инструмента для консоли: не весь JSON, а суть."""
    if "error" in response:                       # инструмент вернул ошибку-наблюдение
        return f"ошибка: {response['error']}"
    if "places" in response:                      # список мест: «3 мест из 4: А, Б, В»
        names = ", ".join(p["name"] for p in response["places"][:4])   # первые 4 названия
        total = response.get("found_total")       # есть только у search_places
        head = f"{len(response['places'])} мест" + (f" из {total}" if total is not None else "")
        return f"{head}: {names}{'…' if len(response['places']) > 4 else ''}"
    return json.dumps(response, ensure_ascii=False)[:200]   # остальное — JSON, обрезанный до 200 символов


async def ask(runner: Runner, session_id: str, question: str, show_thoughts: bool) -> None:
    """Задать один вопрос в сессии и напечатать поток событий агента."""
    settings = get_settings()
    print(f"\n{'═' * 80}\n❓ {question}")
    # usage — счётчик токенов; started — время старта; stream — что сейчас печатаем потоком
    # ("thought" / "text" / None), чтобы ставить подпись только при смене потока.
    usage, started, stream = Usage(), time.perf_counter(), None
    # Сообщение пользователя в формате ADK (google.genai): Content из одной текстовой Part.
    message = types.Content(role="user", parts=[types.Part(text=question)])
    # StreamingMode.SSE — текст и размышления приходят кусками (partial=True);
    # max_llm_calls — предохранитель от зацикливания (урок задачи 6 практики).
    config = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=settings.max_llm_calls)

    # run_async — асинхронный генератор: отдаёт события по мере того, как агент работает.
    # Один вопрос = несколько шагов цикла: запрос к LLM → вызов инструмента → снова LLM …
    async for event in runner.run_async(user_id="console", session_id=session_id,
                                        new_message=message, run_config=config):
        usage.add(event)                                  # учесть токены, если они в событии есть
        # У события может быть content с несколькими частями (Part). Каждая часть — что-то одно:
        # текст (обычный или размышление), вызов инструмента или его результат.
        for part in (event.content.parts if event.content else []):
            if part.text and event.partial:
                # Кусок потока. part.thought=True — это размышление модели, иначе — текст ответа.
                kind = "thought" if part.thought else "text"
                if kind == "thought" and not show_thoughts:
                    continue                              # флаг --no-thoughts: размышления пропускаем
                if kind != stream:   # сменился поток — новая строка с подписью
                    print(f"\n{DIM}💭 " if kind == "thought" else f"{RESET}\n✍️  ", end="")
                    stream = kind
                print(part.text, end="", flush=True)      # flush — показать сразу, без буферизации
            elif part.function_call:
                # Модель попросила вызвать инструмент: имя и аргументы (уже dict, не строка).
                args = json.dumps(part.function_call.args, ensure_ascii=False)
                print(f"{RESET}\n🔧 {part.function_call.name}({args})")
                stream = None
            elif part.function_response:
                # ADK выполнил инструмент; response — тот самый dict, что вернула наша функция.
                print(f"📦 {short_result(part.function_response.response)}")
                stream = None
            # Итоговые (не partial) события с текстом тут не печатаем: они повторяют
            # уже показанный поток целиком — ADK присылает их, чтобы сохранить в историю.

    print(RESET)
    u = usage.as_dict(settings)
    print(f"── запросов к LLM: {u['llm_calls']} · токены: вход {u['input_tokens']}, "
          f"(из кэша {u['cached_tokens']}), выход {u['output_tokens']} · цена ≈ {u['cost_rub']} ₽ · "
          f"{time.perf_counter() - started:.1f} с")


async def main(questions: list[str], show_thoughts: bool) -> None:
    settings = get_settings()
    if not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY пуст: впиши ключ в ai_course_agents/.env")
    # Сервис сессий хранит историю диалога. InMemory — в памяти: для консоли этого хватает,
    # в веб-сервисе (этап 3) будет SqliteSessionService — история переживёт перезапуск.
    sessions = InMemorySessionService()
    # Runner связывает агента и хранилище сессий; app_name — пространство имён для сессий.
    runner = Runner(app_name=APP_NAME, agent=build_agent(settings), session_service=sessions)
    # Одна сессия на все вопросы из командной строки — поэтому работают уточнения.
    session = await sessions.create_session(app_name=APP_NAME, user_id="console")
    for q in questions:
        await ask(runner, session.id, q, show_thoughts)


# Этот блок выполняется только при запуске файла как программы (python -m ...),
# но не при импорте модуля из другого кода.
if __name__ == "__main__":
    args = sys.argv[1:]                                  # всё, что написано после имени модуля
    show = "--no-thoughts" not in args
    # Вопросы — все аргументы, не начинающиеся с «--»; без вопросов — вопрос по умолчанию.
    qs = [a for a in args if not a.startswith("--")] or ["Сколько в Ростове круглосуточных аптек?"]
    asyncio.run(main(qs, show))                          # запустить цикл событий и дождаться main()
