"""Разовое обогащение описаний заведений через DeepSeek (CLI).

    uv run python -m ai_course_agents.search.enrich --sample 10   # показать 10 догадок, ничего не сохранять
    uv run python -m ai_course_agents.search.enrich --all         # все заведения → data/enrichment.json

Проблема: у двух третей заведений общепита в базе нет тега кухни — только название.
«Хинкальная», «Суши-мастер», «Осетинские пироги» сами говорят, что там едят, но модель
эмбеддингов из одного названия этого не вытащит так же уверенно, как языковая модель.
Поэтому один раз просим LLM по названию и виду заведения предположить кухню и блюда.

Честность данных:
  • это ДОГАДКА, а не факт: в файле у каждой записи source="llm_guess", а в тексте
    описания она идёт после слова «предположительно»;
  • модели разрешено ответить null («по названию не понять»: «Орхидея», «Дар») —
    лучше пустое поле, чем выдуманная кухня;
  • результат кэшируется в data/enrichment.json и коммитится в git: сервис при запуске
    LLM для этого не зовёт, а повторный --all досчитывает только новые места.
"""

from __future__ import annotations

import argparse          # разбор аргументов командной строки (--sample / --all)
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone

import litellm           # тот же клиент, через который ADK ходит в DeepSeek

from ai_course_agents.config import get_settings
from ai_course_agents.poi.kinds import FOOD_KINDS, KIND_LABELS
from ai_course_agents.poi.repo import PoiRepo

BATCH = 50               # мест в одном запросе: меньше запросов, но ответ ещё помещается целиком
PARALLEL = 3             # сколько пачек отправлять одновременно — не упираться в лимиты API

SYSTEM = """\
Ты помогаешь составить описания заведений общепита Ростова-на-Дону для поиска по смыслу.
По названию и виду заведения предположи, что там, вероятно, подают: кухню и 3–6 типичных
блюд или напитков. Пиши по-русски, коротко, через запятую, без лишних слов.

Если по названию нельзя уверенно понять кухню (например, «Орхидея», «Дар», «1984»),
верни null — пустое поле лучше случайной догадки. Вид заведения можно учитывать
(бар — напитки, кондитерская — торты), но не придумывай конкретику без оснований.

Ответь JSON-объектом: {"items": [{"id": <id>, "guess": "<строка>" или null}, ...]}
для КАЖДОГО заведения из списка."""


async def candidates(repo: PoiRepo) -> list[dict]:
    """Заведения общепита с названием, но без тега кухни — им и нужна догадка."""
    rows = await repo.named_places(FOOD_KINDS)
    return [r for r in rows if r["name"] and not r["cuisine"]]


async def guess_batch(batch: list[dict], model: str, api_key: str) -> dict[int, str | None]:
    """Один запрос к LLM на пачку заведений → {id: догадка или None}."""
    # Список для модели: одна строка на заведение, «id | вид | название».
    lines = "\n".join(f"{p['id']} | {KIND_LABELS.get(p['kind'], p['kind'])} | {p['name']}"
                      for p in batch)
    response = await litellm.acompletion(
        model=model, api_key=api_key,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Заведения (id | вид | название):\n{lines}"}],
        # json_object (в отличие от json_schema) DeepSeek поддерживает: гарантирует,
        # что ответ — валидный JSON; форму объекта задаём в промпте и проверяем ниже.
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    ids = {p["id"] for p in batch}
    result: dict[int, str | None] = {}
    for item in data.get("items", []):
        pid = int(item.get("id", -1))
        if pid in ids:                                   # чужие/выдуманные id отбрасываем
            guess = item.get("guess")
            result[pid] = guess.strip() if isinstance(guess, str) and guess.strip() else None
    return result


async def run(sample: int | None) -> None:
    settings = get_settings()
    if not settings.deepseek_api_key:
        sys.exit("DEEPSEEK_API_KEY пуст: впиши ключ в ai_course_agents/.env")
    repo = PoiRepo(settings.poi_db)
    places = await candidates(repo)
    path = settings.enrichment_file

    # Уже посчитанное читаем из файла, чтобы не платить дважды.
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": {}}
    if sample:
        todo = random.sample(places, sample)             # случайные N — для оценки качества
    else:
        todo = [p for p in places if str(p["id"]) not in store["items"]]
    print(f"заведений без кухни: {len(places)}, к обработке: {len(todo)}")

    started = time.perf_counter()
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]   # нарезка по BATCH
    sem = asyncio.Semaphore(PARALLEL)                    # не больше PARALLEL запросов одновременно

    async def one(batch: list[dict]) -> dict[int, str | None]:
        async with sem:
            return await guess_batch(batch, settings.llm_model, settings.deepseek_api_key)

    # gather запускает все пачки параллельно (семафор ограничивает число одновременных)
    results = await asyncio.gather(*(one(b) for b in batches))
    guesses = {pid: g for r in results for pid, g in r.items()}

    by_id = {p["id"]: p for p in todo}
    if sample:
        for p in todo:                                   # режим просмотра: печатаем и выходим
            print(f"- {p['name']} ({KIND_LABELS.get(p['kind'])}) → {guesses.get(p['id'])}")
        print(f"\n{time.perf_counter() - started:.1f} с; в файл ничего не записано")
        return

    for pid, guess in guesses.items():
        store["items"][str(pid)] = {"name": by_id[pid]["name"], "guess": guess,
                                    "source": "llm_guess"}
    store["model"] = settings.llm_model
    store["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    # indent=1 и ensure_ascii=False — файл читается глазами и даёт аккуратный diff в git
    path.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    filled = sum(1 for v in store["items"].values() if v["guess"])
    print(f"готово за {time.perf_counter() - started:.1f} с: {len(store['items'])} записей, "
          f"с догадкой {filled}, без догадки (null) {len(store['items']) - filled} → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)   # ровно один из двух флагов
    mode.add_argument("--sample", type=int, help="показать догадки для N случайных заведений")
    mode.add_argument("--all", action="store_true", help="обработать все и сохранить")
    args = parser.parse_args()
    asyncio.run(run(args.sample))
