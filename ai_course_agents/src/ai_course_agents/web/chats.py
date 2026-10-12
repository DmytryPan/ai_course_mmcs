"""Чаты пользователя: список, создание, переименование, удаление, история. Маршруты /api/chats/*.

Каждый чат — это пара «строка в нашей таблице chats» + «сессия ADK» с тем же id.
Все маршруты требуют входа (Depends(current_user)) и работают только с чатами
этого пользователя: чужой id → 404.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ai_course_agents.agent.agent import APP_NAME
from ai_course_agents.web.deps import Services, current_user, get_services
from ai_course_agents.web.events import build_history

router = APIRouter(prefix="/api/chats", tags=["chats"])

DEFAULT_TITLE = "Новый чат"


class ChatPatch(BaseModel):
    title: str = Field(min_length=1, max_length=80)


async def get_owned_chat(chat_id: str, user: dict, services: Services) -> dict:
    """Чат, если он принадлежит пользователю, иначе 404. Используется и в stream.py."""
    chat = await services.db.get_chat(chat_id, user["id"])
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "чат не найден")
    return chat


@router.get("")
async def list_chats(user: dict = Depends(current_user),
                     services: Services = Depends(get_services)) -> list[dict]:
    """Чаты пользователя, свежие сверху — для боковой панели."""
    return await services.db.list_chats(user["id"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat(user: dict = Depends(current_user),
                      services: Services = Depends(get_services)) -> dict:
    """Новый пустой чат: сессия ADK + строка в таблице chats с тем же id."""
    chat_id = uuid.uuid4().hex                        # 32 случайных hex-символа: id не угадать
    # user_id в ADK — строка; берём наш числовой id. Сессия «принадлежит» этому пользователю:
    # get_session с другим user_id её не найдёт.
    await services.sessions.create_session(app_name=APP_NAME, user_id=str(user["id"]),
                                           session_id=chat_id)
    return await services.db.create_chat(chat_id, user["id"], DEFAULT_TITLE)


@router.patch("/{chat_id}")
async def rename_chat(chat_id: str, body: ChatPatch, user: dict = Depends(current_user),
                      services: Services = Depends(get_services)) -> dict:
    await get_owned_chat(chat_id, user, services)
    await services.db.update_chat(chat_id, user["id"], title=body.title.strip())
    return await get_owned_chat(chat_id, user, services)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str, user: dict = Depends(current_user),
                      services: Services = Depends(get_services)) -> None:
    """Удалить чат и его историю в ADK. Пока в чате идёт ответ — 409, удалять нельзя."""
    await get_owned_chat(chat_id, user, services)
    if services.chat_lock(chat_id).locked():
        raise HTTPException(status.HTTP_409_CONFLICT, "в чате идёт ответ — дождитесь окончания")
    await services.sessions.delete_session(app_name=APP_NAME, user_id=str(user["id"]),
                                           session_id=chat_id)
    await services.db.delete_chat(chat_id, user["id"])
    services.chat_locks.pop(chat_id, None)            # замок удалённого чата больше не нужен


@router.get("/{chat_id}/messages")
async def chat_history(chat_id: str, user: dict = Depends(current_user),
                       services: Services = Depends(get_services)) -> dict:
    """История чата для интерфейса: сообщения, шаги агента и места для карты."""
    chat = await get_owned_chat(chat_id, user, services)
    session = await services.sessions.get_session(app_name=APP_NAME, user_id=str(user["id"]),
                                                  session_id=chat_id)
    events = session.events if session else []
    return {"chat": chat, "messages": build_history(events)}
