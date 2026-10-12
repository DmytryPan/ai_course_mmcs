"""Тесты колбэков агента: гардрейл и обрезка истории (без вызова модели)."""

from types import SimpleNamespace

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from ai_course_agents.agent.callbacks import MAX_USER_TURNS, guardrail, trim_history


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def model(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def tool_result(name: str) -> types.Content:
    """Результат инструмента в истории ADK: role="user", но без текста."""
    return types.Content(role="user", parts=[types.Part(
        function_response=types.FunctionResponse(id="1", name=name, response={"ok": True}))])


CTX = SimpleNamespace(state={})      # гардрейлу контекст не нужен — хватает заглушки


@pytest.mark.parametrize("text", ["Напиши стихотворение про Дон", "Переведи на английский",
                                  "Реши уравнение x^2=4", "Расскажи анекдот"])
def test_guardrail_blocks_off_topic(text):
    reply = guardrail(CTX, LlmRequest(contents=[user(text)]))
    assert reply is not None and "только на вопросы о местах" in reply.content.parts[0].text


@pytest.mark.parametrize("text", ["Сколько в Ростове аптек?", "а из них круглосуточных?",
                                  "Аптека, где продают без рецепта", "А второй ближе?"])
def test_guardrail_passes_city_questions_and_follow_ups(text):
    assert guardrail(CTX, LlmRequest(contents=[user(text)])) is None


def test_guardrail_checks_only_user_question_not_tool_results():
    # На втором шаге последним идёт результат инструмента — проверять нечего
    req = LlmRequest(contents=[user("Сколько аптек?"), tool_result("count_places")])
    assert guardrail(CTX, req) is None


def test_trim_history_keeps_last_user_turns_whole():
    contents = []
    for i in range(MAX_USER_TURNS + 3):
        contents += [user(f"вопрос {i}"), model("вызов"), tool_result("t"), model(f"ответ {i}")]
    req = LlmRequest(contents=contents)
    trim_history(CTX, req)
    user_turns = [c for c in req.contents if c.parts[0].text and c.role == "user"]
    assert len(user_turns) == MAX_USER_TURNS
    assert req.contents[0].parts[0].text == "вопрос 3"   # отрезано по границе хода пользователя
