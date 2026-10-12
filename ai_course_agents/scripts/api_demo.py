"""Проверка бэкенда через HTTP — контрольная точка этапа 3 (сервер должен быть запущен).

    uv run ai-course-agents                      # в одном терминале
    uv run python scripts/api_demo.py            # в другом

Сценарий:
  1. два пользователя регистрируются, у каждого свои cookie;
  2. alice создаёт чат и задаёт вопрос — печатаем поток SSE по событиям;
  3. уточняющий вопрос в том же чате (учёт контекста);
  4. посторонняя просьба — гардрейл отвечает без вызова LLM;
  5. bob пытается прочитать чат alice — должен получить 404;
  6. alice и bob спрашивают ОДНОВРЕМЕННО — оба потока идут параллельно.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


async def user(name: str) -> httpx.AsyncClient:
    """Клиент со своими cookie: регистрируем пользователя с уникальным логином."""
    client = httpx.AsyncClient(base_url=BASE, timeout=180)
    login = f"{name}_{uuid.uuid4().hex[:6]}"
    r = await client.post("/api/auth/register", json={"username": login, "password": "secret123"})
    r.raise_for_status()
    print(f"✓ {login}: зарегистрирован, cookie получен")
    return client


async def ask(client: httpx.AsyncClient, chat_id: str, text: str, tag: str = "") -> dict:
    """Задать вопрос и печатать поток SSE: шаги — строкой, текст — склеенным."""
    print(f"\n{tag}❓ {text}")
    started, first_event, kind, done = time.perf_counter(), None, None, {}
    chunks = {"thinking": 0, "text": 0}
    async with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"text": text}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():               # SSE — текстовые строки
            if line.startswith("event:"):
                kind = line[6:].strip()
                first_event = first_event or time.perf_counter() - started
            elif line.startswith("data:") and kind:
                data = json.loads(line[5:].strip())
                if kind in chunks:
                    chunks[kind] += 1
                elif kind == "tool_call":
                    print(f"{tag}  🔧 {data['title']}: {json.dumps(data['args'], ensure_ascii=False)}")
                elif kind == "tool_result":
                    print(f"{tag}  📦 {data['summary']} · {data['ms']} мс · точек для карты: {len(data['places'])}")
                elif kind == "done":
                    done = data
                elif kind in ("error", "title", "queued"):
                    print(f"{tag}  [{kind}] {data}")
    u = done.get("usage", {})
    print(f"{tag}  ✍️  {done.get('answer', '')[:300]}")
    print(f"{tag}  ── первое событие через {first_event:.2f} с; кусков: размышления {chunks['thinking']}, "
          f"ответ {chunks['text']}; LLM-запросов {u.get('llm_calls')}, {u.get('cost_rub')} ₽, "
          f"{done.get('ms')} мс, гардрейл: {done.get('guardrail')}")
    return done


async def main() -> None:
    alice, bob = await user("alice"), await user("bob")
    chat = (await alice.post("/api/chats")).json()
    print(f"✓ alice создала чат {chat['id']}")

    await ask(alice, chat["id"], "Какие кафе есть на Пушкинской улице?")
    await ask(alice, chat["id"], "А какое из них ближе всего к Большой Садовой, 50?")
    await ask(alice, chat["id"], "Напиши стихотворение про Дон")

    r = await bob.get(f"/api/chats/{chat['id']}/messages")
    print(f"\n{'✓' if r.status_code == 404 else '✗'} bob читает чат alice → {r.status_code}")
    r = await bob.post(f"/api/chats/{chat['id']}/messages", json={"text": "привет"})
    print(f"{'✓' if r.status_code == 404 else '✗'} bob пишет в чат alice → {r.status_code}")

    history = (await alice.get(f"/api/chats/{chat['id']}/messages")).json()
    titles = [c["title"] for c in (await alice.get("/api/chats")).json()]
    print(f"✓ история чата alice: {len(history['messages'])} сообщений; чаты alice: {titles}")

    print("\n── одновременные вопросы alice и bob ──")
    chat_b = (await bob.post("/api/chats")).json()
    t = time.perf_counter()
    await asyncio.gather(ask(alice, chat["id"], "Сколько в Ростове круглосуточных аптек?", "[alice] "),
                         ask(bob, chat_b["id"], "Где поесть пельменей?", "[bob] "))
    print(f"\n✓ оба ответа за {time.perf_counter() - t:.1f} с (параллельно, а не друг за другом)")
    await alice.aclose(); await bob.aclose()


if __name__ == "__main__":
    asyncio.run(main())
