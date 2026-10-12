"""База приложения (data/app.sqlite): пользователи и список их чатов.

История сообщений сюда НЕ пишется — её хранит ADK в своей базе сессий (data/sessions.sqlite).
Здесь только то, чего у ADK нет: учётные записи и «оглавление» чатов (название, время).
Чат в нашей базе и сессия ADK связаны одним id: chat.id == session.id.

Почему отдельная таблица чатов, а не list_sessions из ADK: ADK отдаёт сессии без
человеческого названия и без сортировки по последней активности, а боковой панели
интерфейса нужны именно они.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,   -- «Dima» и «dima» — один и тот же логин
    password_hash TEXT NOT NULL,                         -- bcrypt-хеш, сам пароль нигде не хранится
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    id         TEXT PRIMARY KEY,                         -- он же id сессии ADK
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at DESC);
"""


class AppDB:
    """Тонкая обёртка над aiosqlite. Соединение на запрос — как в poi/repo.py."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def init(self) -> None:
        """Создать файл и таблицы, если их нет (вызывается при старте приложения)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as con:
            await con.executescript(SCHEMA)
            await con.commit()

    async def _execute(self, sql: str, params: tuple = ()) -> int:
        """Изменяющий запрос; возвращает id вставленной строки (для INSERT)."""
        async with aiosqlite.connect(self.path) as con:
            await con.execute("PRAGMA foreign_keys = ON")     # включить каскадное удаление чатов
            cur = await con.execute(sql, params)
            await con.commit()
            return cur.lastrowid

    async def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.path) as con:
            con.row_factory = aiosqlite.Row
            async with con.execute(sql, params) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = await self._rows(sql, params)
        return rows[0] if rows else None

    # ── пользователи ────────────────────────────────────────────────────────

    async def create_user(self, username: str, password_hash: str) -> int:
        """Новый пользователь → его id. Занятый логин даст aiosqlite.IntegrityError (UNIQUE)."""
        return await self._execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, time.time()))

    async def user_by_name(self, username: str) -> dict[str, Any] | None:
        return await self._one("SELECT * FROM users WHERE username = ?", (username,))

    async def user_by_id(self, user_id: int) -> dict[str, Any] | None:
        return await self._one("SELECT id, username FROM users WHERE id = ?", (user_id,))

    # ── чаты ────────────────────────────────────────────────────────────────
    # Во всех запросах к чатам есть условие user_id = ?: даже зная чужой id чата,
    # пользователь не прочитает, не переименует и не удалит его — запрос просто
    # не найдёт строку, и API ответит 404 (не 403: не подтверждаем, что такой чат есть).

    async def create_chat(self, chat_id: str, user_id: int, title: str) -> dict[str, Any]:
        now = time.time()
        await self._execute(
            "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, user_id, title, now, now))
        return {"id": chat_id, "title": title, "created_at": now, "updated_at": now}

    async def list_chats(self, user_id: int) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT id, title, created_at, updated_at FROM chats WHERE user_id = ? "
            "ORDER BY updated_at DESC", (user_id,))

    async def get_chat(self, chat_id: str, user_id: int) -> dict[str, Any] | None:
        return await self._one(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id))

    async def update_chat(self, chat_id: str, user_id: int, *, title: str | None = None) -> None:
        """Обновить время активности (и название, если передано)."""
        if title is None:
            await self._execute("UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?",
                                (time.time(), chat_id, user_id))
        else:
            await self._execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (title, time.time(), chat_id, user_id))

    async def delete_chat(self, chat_id: str, user_id: int) -> None:
        await self._execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
