"""Пароли и токены входа.

Пароли: bcrypt — медленная (нарочно) хеш-функция с солью. Медленная — чтобы перебор
украденной базы хешей был дорогим; соль — чтобы одинаковые пароли давали разные хеши.
Сам пароль нигде не сохраняется и не логируется.

Токен входа: JWT (JSON Web Token) — строка вида «заголовок.данные.подпись». В данных —
id пользователя и срок годности; подпись HMAC-SHA256 секретом сервера (settings.jwt_secret).
Подделать токен без секрета нельзя, поэтому сервер верит id из токена, не храня сессий входа.

Токен кладётся в cookie с флагами:
  • HttpOnly — JavaScript страницы не может его прочитать (XSS не украдёт токен);
  • SameSite=Lax — браузер не отправит cookie с POST-запросом с чужого сайта (защита от CSRF);
  • Secure — только по HTTPS; включается настройкой, т.к. локально сервис работает по HTTP.
"""

from __future__ import annotations

import time

import bcrypt
import jwt                      # пакет PyJWT

COOKIE_NAME = "rg_token"
ALGORITHM = "HS256"             # HMAC-SHA256: один общий секрет для подписи и проверки


def hash_password(password: str) -> str:
    """Пароль → bcrypt-хеш (строка вида $2b$12$...). gensalt() — новая случайная соль."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Совпадает ли пароль с хешем. checkpw сам извлекает соль из хеша."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))


def make_token(user_id: int, secret: str, ttl_hours: int) -> str:
    """Подписанный токен: sub — чей он, iat — когда выдан, exp — до какого момента действует."""
    now = int(time.time())
    payload = {"sub": str(user_id), "iat": now, "exp": now + ttl_hours * 3600}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def read_token(token: str, secret: str) -> int | None:
    """id пользователя из токена или None, если токен поддельный, испорчен или просрочен.

    algorithms=[ALGORITHM] — принимаем только наш алгоритм: иначе токен с alg="none"
    (без подписи) мог бы пройти проверку — классическая уязвимость JWT.
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        return None
