"""Регистрация, вход, выход, «кто я». Маршруты /api/auth/*."""

from __future__ import annotations

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from ai_course_agents.web.deps import Services, current_user, get_services
from ai_course_agents.web.security import (COOKIE_NAME, hash_password, make_token,
                                           verify_password)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    """Тело запроса регистрации и входа. Pydantic проверит его до вызова обработчика:
    не тот тип или длина → FastAPI сам ответит 422 с описанием ошибки."""

    # pattern — только буквы (латиница и кириллица), цифры и . _ -: без пробелов и спецсимволов
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-zА-Яа-яЁё0-9._-]+$")
    # bcrypt учитывает только первые 72 байта пароля: длиннее — запрещаем, а не режем молча
    password: str = Field(min_length=6, max_length=72)


def _login_response(response: Response, user: dict, services: Services) -> dict:
    """Выдать токен в cookie и вернуть публичные данные пользователя."""
    s = services.settings
    token = make_token(user["id"], s.jwt_secret, s.jwt_ttl_hours)
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=s.jwt_ttl_hours * 3600,     # cookie живёт столько же, сколько токен
        httponly=True, samesite="lax",       # см. web/security.py
        secure=s.cookie_secure,              # True — только по HTTPS (на боевом сервере)
        path="/")
    return {"id": user["id"], "username": user["username"]}


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: Credentials, response: Response,
                   services: Services = Depends(get_services)) -> dict:
    """Новый пользователь; сразу же и входит (cookie с токеном в ответе)."""
    try:
        user_id = await services.db.create_user(body.username, hash_password(body.password))
    except aiosqlite.IntegrityError:          # UNIQUE(username) — логин уже занят
        raise HTTPException(status.HTTP_409_CONFLICT, "такой логин уже занят")
    return _login_response(response, {"id": user_id, "username": body.username}, services)


@router.post("/login")
async def login(body: Credentials, response: Response,
                services: Services = Depends(get_services)) -> dict:
    user = await services.db.user_by_name(body.username)
    # Одинаковое сообщение для «нет такого логина» и «неверный пароль»: не подсказываем,
    # какие логины существуют.
    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "неверный логин или пароль")
    return _login_response(response, user, services)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    """Выход — просто удалить cookie. Сам токен до истечения срока остаётся валидным:
    для MVP это осознанное упрощение (иначе нужен список отозванных токенов)."""
    response.delete_cookie(COOKIE_NAME, path="/")


@router.get("/me")
async def me(user: dict = Depends(current_user)) -> dict:
    """Кто вошёл. Фронтенд зовёт при загрузке страницы: 401 → показать экран входа."""
    return user
