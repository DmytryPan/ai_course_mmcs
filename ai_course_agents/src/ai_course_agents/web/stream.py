"""Ответ агента потоком SSE: POST /api/chats/{id}/messages.

SSE (Server-Sent Events) — ответ HTTP, который не закрывается сразу, а присылает
события по мере готовности. Формат — текст:  «event: <тип>\\ndata: <JSON>\\n\\n».
Браузер видит шаги агента в реальном времени: размышления, вызовы инструментов,
их результаты и сам ответ, печатающийся по кусочкам.

Типы событий (поле event), данные — JSON:
  start       {chat_id}                           — запрос принят
  title       {title}                              — чату дано название (по первому вопросу)
  queued      {}                                   — все слоты LLM заняты, ждём очереди
  thinking    {text}                               — кусок размышлений модели (DeepSeek: по-английски)
  text        {text}                               — кусок ответа пользователю
  tool_call   {id, name, title, args}              — агент вызвал инструмент
  tool_result {id, name, summary, error, places, ms} — результат инструмента (+ точки для карты)
  done        {answer, usage, ms, guardrail}       — ответ готов: итоговый текст, токены, цена
  error       {message, retryable}                 — сбой: понятный текст и можно ли повторить

Если текст пришёл, а следом — tool_call, это был не ответ, а «пояснение перед шагом»
(«сейчас найду…»): фронтенд переносит такой текст в «Ход мысли».

Почему POST, а не GET с EventSource: вопрос пользователя — это данные в теле запроса,
а встроенный в браузер EventSource умеет только GET. Фронтенд читает поток через fetch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncIterator

import anyio
from fastapi import APIRouter, Depends, HTTPException, status
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ai_course_agents.agent.agent import APP_NAME
from ai_course_agents.agent.usage import Usage
from ai_course_agents.web.chats import DEFAULT_TITLE, get_owned_chat
from ai_course_agents.web.deps import Services, current_user, get_services
from ai_course_agents.web.events import TOOL_TITLES, summarize_tool_result

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chats", tags=["stream"])


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=1000)   # длинные «простыни» не принимаем


def sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    """Одно SSE-событие в формате sse-starlette. ensure_ascii=False — кириллица как есть."""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


# ── ошибки → понятные сообщения ─────────────────────────────────────────────

def _error_chain(exc: BaseException) -> list[BaseException]:
    """Исключение и все его «причины». ADK заворачивает ошибку модели в свои исключения
    (DynamicNodeFailError → ... → litellm.APIConnectionError), поэтому смотрим всю цепочку."""
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__ or getattr(exc, "error", None)
        if not isinstance(exc, BaseException):
            break
    return chain


def explain_error(exc: BaseException) -> tuple[str, bool]:
    """(сообщение для пользователя, можно ли просто повторить)."""
    for e in _error_chain(exc):
        name = type(e).__name__
        if isinstance(e, LlmCallsLimitExceededError):
            return ("Агент сделал слишком много шагов и был остановлен предохранителем. "
                    "Попробуйте спросить проще или точнее.", False)
        if "Authentication" in name:
            return "Сервис модели не принял ключ API. Проверьте DEEPSEEK_API_KEY в .env.", False
        if "RateLimit" in name:
            return "Сервис модели перегружен запросами. Попробуйте через минуту.", True
        if "Timeout" in name:
            return "Модель слишком долго не отвечала. Попробуйте ещё раз.", True
        if any(k in name for k in ("Connection", "MidStream", "ReadError", "ServiceUnavailable",
                                   "InternalServerError")):
            return "Соединение с моделью прервалось. Повторите вопрос.", True
    return "Внутренняя ошибка сервиса. Попробуйте ещё раз; если повторится — сообщите нам.", True


# ── журнал прогонов ─────────────────────────────────────────────────────────

def _write_run_log(services: Services, record: dict[str, Any]) -> None:
    """Одна строка JSON на ответ в data/runs/ГГГГ-ММ-ДД.jsonl (формат JSON Lines).
    По этим строкам считаются токены, цена, частые ошибки и шаги агента."""
    runs = services.settings.runs_dir
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{datetime.now():%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as f:              # "a" — дописать в конец файла
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _user_hash(user_id: int, secret: str) -> str:
    """В журнал — не id пользователя, а его хеш с секретом: строки одного пользователя
    можно сгруппировать, но связать с учётной записью без секрета нельзя."""
    return hashlib.sha256(f"{secret}:{user_id}".encode()).hexdigest()[:12]


# ── закрытие прерванного хода ───────────────────────────────────────────────

async def close_interrupted_turn(services: Services, user_id: int, chat_id: str,
                                 invocation_id: str, note: str) -> None:
    """Дописать в историю сессии, что ответ прерван (отмена пользователем или ошибка).

    Зачем: вопрос пользователя ADK сохраняет сразу, а ответа после обрыва нет. На следующем
    вопросе модель увидит «висящий» прошлый вопрос и ответит на оба (так и было при проверке:
    «Сколько аптек?» после остановленного «Где поесть шаурмы?» запустил ещё и поиск шаурмы).
    Плюс обрыв между вызовом инструмента и его результатом оставил бы в истории
    function_call без function_response — такой запрос DeepSeek отвергнет с ошибкой 400.
    Поэтому: на каждый «висящий» вызов дописываем результат-ошибку, затем реплику агента.
    """
    session = await services.sessions.get_session(app_name=APP_NAME, user_id=str(user_id),
                                                  session_id=chat_id)
    if session is None:
        return
    agent_name = services.runner.agent.name
    calls, answered = {}, set()                       # вызовы этого ответа и те, на что есть результат
    for e in session.events:
        if e.invocation_id != invocation_id or not e.content:
            continue
        for part in e.content.parts or []:
            if part.function_call:
                calls[part.function_call.id] = part.function_call.name
            if part.function_response:
                answered.add(part.function_response.id)
    dangling = [types.Part(function_response=types.FunctionResponse(
        id=cid, name=name, response={"error": "вызов прерван"}))
        for cid, name in calls.items() if cid not in answered]
    if dangling:
        # Результаты инструментов в истории ADK идут с role="user" и автором-агентом.
        await services.sessions.append_event(session, Event(
            invocation_id=invocation_id, author=agent_name,
            content=types.Content(role="user", parts=dangling)))
    await services.sessions.append_event(session, Event(
        invocation_id=invocation_id, author=agent_name,
        content=types.Content(role="model", parts=[types.Part(text=note)])))


# ── сам поток ───────────────────────────────────────────────────────────────

async def run_agent(services: Services, user: dict, chat: dict,
                    text: str) -> AsyncIterator[dict[str, str]]:
    """Генератор SSE-событий одного ответа агента."""
    s = services.settings
    chat_id, user_id = chat["id"], user["id"]
    lock = services.chat_lock(chat_id)
    if lock.locked():                     # в этом чате уже идёт ответ (повторное нажатие)
        yield sse("error", {"message": "Предыдущий ответ ещё не готов.", "retryable": False})
        return

    started = time.perf_counter()
    usage = Usage()
    record: dict[str, Any] = {"ts": datetime.now().isoformat(timespec="seconds"),
                              "user": _user_hash(user_id, s.jwt_secret), "chat": chat_id,
                              "question": text, "tools": [], "status": "ok"}
    async with lock:
        yield sse("start", {"chat_id": chat_id})
        if chat["title"] == DEFAULT_TITLE:                      # первый вопрос → название чата
            title = text if len(text) <= 60 else text[:57].rstrip() + "…"
            await services.db.update_chat(chat_id, user_id, title=title)
            yield sse("title", {"title": title})

        if services.llm_slots.locked():                         # все слоты заняты — сообщим
            yield sse("queued", {})
        answer = ""
        invocation_id = ""                                      # id этого прогона в ADK (из событий)
        tool_started: dict[str, float] = {}                     # id вызова → время старта
        try:
            async with services.llm_slots:                      # ждём свободный слот LLM
                message = types.Content(role="user", parts=[types.Part(text=text)])
                config = RunConfig(streaming_mode=StreamingMode.SSE,
                                   max_llm_calls=s.max_llm_calls)
                async for event in services.runner.run_async(
                        user_id=str(user_id), session_id=chat_id,
                        new_message=message, run_config=config,
                        # Первым событием ADK вернёт сам вопрос пользователя — так мы сразу
                        # узнаём invocation_id и сможем закрыть ход, даже если отмена придёт
                        # раньше первого ответа модели.
                        yield_user_message=True):
                    invocation_id = event.invocation_id or invocation_id
                    if event.author == "user":                  # это сам вопрос — в поток не шлём
                        continue
                    usage.add(event)
                    parts = event.content.parts if event.content and event.content.parts else []
                    for p in parts:
                        if p.text and event.partial:            # кусок потока
                            yield sse("thinking" if p.thought else "text", {"text": p.text})
                        elif p.function_call:
                            fc = p.function_call
                            tool_started[fc.id] = time.perf_counter()
                            args = dict(fc.args or {})
                            record["tools"].append({"name": fc.name, "args": args})
                            yield sse("tool_call", {"id": fc.id, "name": fc.name,
                                                    "title": TOOL_TITLES.get(fc.name, fc.name),
                                                    "args": args})
                        elif p.function_response:
                            fr = p.function_response
                            info = summarize_tool_result(fr.response)
                            ms = int((time.perf_counter() - tool_started.pop(fr.id, started)) * 1000)
                            if info["error"]:
                                record["tools"][-1]["error"] = info["error"]
                            yield sse("tool_result", {"id": fr.id, "name": fr.name, "ms": ms, **info})
                        elif p.text and not p.thought:
                            # Итоговое (не partial) событие: тот же текст целиком — ADK присылает
                            # его после кусков, чтобы сохранить в историю. У ответа модели есть
                            # usage_metadata, и куски уже ушли потоком. У ответа гардрейла
                            # usage нет (модель не вызывалась) и кусков не было — отправляем целиком.
                            if event.usage_metadata is None:
                                yield sse("text", {"text": p.text})
                            answer = p.text
            # Ответ дал гардрейл, если за весь ответ не было ни одного запроса к LLM.
            guardrail = usage.llm_calls == 0 and bool(answer)
            record.update(status="blocked" if guardrail else "ok", answer=answer)
            yield sse("done", {"answer": answer, "usage": usage.as_dict(s), "guardrail": guardrail,
                               "ms": int((time.perf_counter() - started) * 1000)})
        except asyncio.CancelledError:
            # Клиент закрыл соединение (кнопка «Стоп» или ушёл со страницы): sse-starlette
            # отменяет генератор, отмена прерывает и прогон агента. Пробрасываем дальше.
            record["status"] = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 — любая ошибка должна стать понятным событием
            message, retryable = explain_error(exc)
            log.exception("ошибка прогона агента в чате %s", chat_id)
            record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
            yield sse("error", {"message": message, "retryable": retryable})
        finally:
            # При отмене (клиент ушёл) anyio отменяет КАЖДЫЙ следующий await в этой задаче —
            # без защиты запись в журнал и обновление чата тоже прервались бы, и отменённый
            # прогон пропал бы из журнала (так и было при первой проверке). shield=True —
            # «доделать эти шаги, даже если задачу уже отменяют».
            with anyio.CancelScope(shield=True):
                if record["status"] in ("cancelled", "error") and invocation_id:
                    note = ("[Ответ остановлен пользователем.]" if record["status"] == "cancelled"
                            else "[Ответ прерван из-за ошибки.]")
                    await close_interrupted_turn(services, user_id, chat_id, invocation_id, note)
                await services.db.update_chat(chat_id, user_id)     # чат поднимается наверх списка
                record["usage"] = usage.as_dict(s)
                record["ms"] = int((time.perf_counter() - started) * 1000)
                await asyncio.to_thread(_write_run_log, services, record)


@router.post("/{chat_id}/messages")
async def ask(chat_id: str, body: Question, user: dict = Depends(current_user),
              services: Services = Depends(get_services)) -> EventSourceResponse:
    chat = await get_owned_chat(chat_id, user, services)      # чужой чат → 404 до начала потока
    if services.chat_lock(chat_id).locked():
        raise HTTPException(status.HTTP_409_CONFLICT, "предыдущий ответ ещё не готов")
    # ping — каждые 15 с служебное событие, чтобы прокси не закрыли «молчащее» соединение,
    # пока модель долго думает.
    return EventSourceResponse(run_agent(services, user, chat, body.text.strip()), ping=15)
