"""Проверка интерфейса в настоящем браузере (Edge) через Playwright: сценарий пользователя + скриншоты.

    uv run ai-course-agents                                                   # сервер — в одном терминале
    uv run --with playwright python scripts/ui_check.py data/ui_shots          # в другом

Playwright не входит в зависимости проекта: `uv run --with playwright` ставит его во временное
окружение. Браузер не скачивается — используется установленный Microsoft Edge (channel="msedge").

Сценарий: экран входа и ошибка проверки полей → регистрация → приветствие → «Где поесть
пельменей?» (снимок во время ответа и после) → клик по названию места → «Ход мысли» →
уточнение по контексту → гардрейл → «Стоп» → перезагрузка (история сохранилась) →
мобильная ширина в тёмной теме: чат, карта, панель чатов. В конце — ошибки консоли браузера.
"""
import sys
import time
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
errors = []


def shot(page, name):
    page.screenshot(path=str(OUT / f"{name}.png"))
    print("shot", name)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, locale="ru-RU")
    page = ctx.new_page()
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    page.goto(BASE)
    page.wait_for_selector("#auth:not([hidden])")
    shot(page, "01_auth")
    page.click("text=Регистрация")
    page.fill("input[name=username]", "ab")
    page.fill("input[name=password]", "123")
    page.click("#auth-submit")
    page.wait_for_timeout(500)
    print("validation error:", page.inner_text("#auth-error"))
    shot(page, "02_auth_error")
    page.fill("input[name=username]", f"dima_{uuid.uuid4().hex[:5]}")
    page.fill("input[name=password]", "secret123")
    page.click("#auth-submit")
    page.wait_for_selector("#app:not([hidden])")
    page.wait_for_timeout(1500)
    shot(page, "03_welcome")

    page.click("text=Где поесть пельменей?")
    page.wait_for_selector(".step", timeout=60000)
    page.wait_for_timeout(800)
    shot(page, "04_streaming")
    page.wait_for_selector(".msg-bot .meta", timeout=120000)
    page.wait_for_timeout(1500)
    shot(page, "05_answer")
    links = page.query_selector_all(".place-link")
    print("place links:", [l.inner_text() for l in links])
    if links:
        links[0].click()
        page.wait_for_timeout(1200)
        shot(page, "06_place_click")
    page.click(".trace > summary")
    page.wait_for_timeout(300)
    shot(page, "07_trace_open")

    page.fill("#input", "А что из этого ближе всего к Пушкинской улице?")
    page.keyboard.press("Enter")
    page.wait_for_function("document.querySelectorAll('.msg-bot .meta').length >= 2", timeout=120000)
    page.wait_for_timeout(1500)
    shot(page, "08_followup")

    page.fill("#input", "Напиши стихотворение про Дон")
    page.keyboard.press("Enter")
    page.wait_for_function("document.querySelectorAll('.msg-bot .meta').length >= 3", timeout=60000)
    page.wait_for_timeout(800)
    shot(page, "09_guardrail")

    # stop button
    page.fill("#input", "Какие кафе есть на Пушкинской улице?")
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)
    page.click("#send")
    page.wait_for_timeout(1500)
    shot(page, "10_stopped")

    # reload: history persists
    page.reload()
    page.wait_for_selector(".msg-bot", timeout=30000)
    page.wait_for_timeout(1500)
    shot(page, "11_reload_history")
    print("chats in sidebar:", [e.inner_text() for e in page.query_selector_all(".chat-item .title")])

    # mobile
    m = browser.new_context(viewport={"width": 390, "height": 844}, locale="ru-RU", storage_state=ctx.storage_state(),
                            color_scheme="dark", is_mobile=True, has_touch=True)
    mp = m.new_page()
    mp.on("pageerror", lambda e: errors.append(f"mobile pageerror: {e}"))
    mp.goto(BASE)
    mp.wait_for_selector(".msg-bot", timeout=30000)
    mp.wait_for_timeout(1500)
    shot(mp, "12_mobile_dark")
    mp.click("#open-map")
    mp.wait_for_timeout(1500)
    shot(mp, "13_mobile_map")
    mp.click("#close-map")
    mp.click("#open-sidebar")
    mp.wait_for_timeout(500)
    shot(mp, "14_mobile_sidebar")
    browser.close()

print("ERRORS:", errors or "none")
