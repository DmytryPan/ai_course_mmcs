"""Справочник видов мест (поле poi.kind) с русскими названиями.

kind в базе — латинское значение тега OpenStreetMap (pharmacy, cafe, outpost …).
Модель должна передавать в инструменты именно его, а пользователь пишет по-русски.
Поэтому:
  • KIND_LABELS — словарь kind → русское название; он попадает в инструкцию агента,
    чтобы модель знала, как перевести «аптека» в pharmacy;
  • KindName — тип Literal из тех же ключей: ADK превращает его в "enum" в JSON Schema
    аргумента, и модель не может передать вид, которого нет в базе (урок задачи 3 практики).

В перечень вошли виды, которых в базе 15 и больше (79 штук), плюс кино, театры и хостелы.
Редкие виды (их ещё ~100, по 1–14 мест) оставлены за бортом: схема с сотнями значений
раздувает каждый запрос к модели, а пользы почти не даёт.
"""

from __future__ import annotations

from typing import Literal

KIND_LABELS: dict[str, str] = {
    # еда и напитки
    "restaurant": "ресторан",
    "cafe": "кафе, кофейня",
    "fast_food": "фастфуд, шаурма, бургерная",
    "bar": "бар",
    "pub": "паб",
    "nightclub": "ночной клуб",
    "bakery": "пекарня",
    "confectionery": "кондитерская",
    # продукты и магазины
    "supermarket": "супермаркет",
    "convenience": "продуктовый магазин у дома",
    "greengrocer": "овощи и фрукты",
    "butcher": "мясной магазин",
    "seafood": "рыба и морепродукты",
    "alcohol": "алкоголь",
    "kiosk": "киоск",
    "marketplace": "рынок",
    "mall": "торговый центр",
    "variety_store": "магазин фиксированных цен, всё для дома",
    "houseware": "товары для дома",
    "clothes": "одежда",
    "shoes": "обувь",
    "jewelry": "ювелирный",
    "cosmetics": "косметика",
    "chemist": "бытовая химия",
    "electronics": "электроника",
    "mobile_phone": "салон связи, телефоны",
    "computer": "компьютеры",
    "furniture": "мебель",
    "hardware": "хозтовары",
    "doityourself": "строительный магазин",
    "power_tools": "электроинструмент",
    "books": "книжный",
    "stationery": "канцтовары",
    "toys": "игрушки",
    "sports": "спорттовары",
    "outdoor": "туризм и активный отдых",
    "pet": "зоомагазин",
    "florist": "цветы",
    "optician": "оптика",
    "tobacco": "табак",
    "e-cigarette": "электронные сигареты, вейпы",
    "outpost": "пункт выдачи заказов (Ozon, Wildberries…)",
    "pawnbroker": "ломбард",
    "bookmaker": "букмекер",
    "travel_agency": "турагентство",
    # авто
    "fuel": "заправка, АЗС",
    "car_wash": "автомойка",
    "car_repair": "автосервис",
    "car_parts": "автозапчасти",
    "tyres": "шины, шиномонтаж",
    "car": "автосалон",
    # здоровье
    "pharmacy": "аптека",
    "hospital": "больница",
    "clinic": "поликлиника, клиника",
    "doctors": "врач, медкабинет",
    "dentist": "стоматология",
    "veterinary": "ветклиника",
    # услуги
    "bank": "банк",
    "atm": "банкомат",
    "post_office": "почта",
    "beauty": "салон красоты",
    "hairdresser": "парикмахерская",
    "lawyer": "юрист",
    "estate_agent": "агентство недвижимости",
    "company": "офис компании",
    # образование и дети
    "kindergarten": "детский сад",
    "school": "школа",
    "college": "колледж",
    "university": "университет",
    "library": "библиотека",
    "playground": "детская площадка",
    # отдых и спорт
    "park": "парк",
    "attraction": "достопримечательность",
    "museum": "музей",
    "cinema": "кинотеатр",
    "theatre": "театр",
    "sports_centre": "спортивный центр",
    "fitness_centre": "фитнес-клуб",
    "swimming_pool": "бассейн",
    # где остановиться
    "hotel": "гостиница",
    "guest_house": "гостевой дом",
    "hostel": "хостел",
}

# Literal[...] из ключей словаря. Literal.__getitem__ принимает кортеж — так перечень
# строится один раз из словаря и не расходится с ним при правках.
KindName = Literal[tuple(KIND_LABELS)]  # type: ignore[valid-type]

# Виды общепита: ими ограничивается семантический поиск «где поесть».
FOOD_KINDS = ("restaurant", "cafe", "fast_food", "bar", "pub", "bakery", "confectionery")


def kinds_cheatsheet() -> str:
    """Компактная шпаргалка «kind — по-русски» для инструкции агента (одна строка на вид)."""
    return "\n".join(f"{kind} — {label}" for kind, label in KIND_LABELS.items())
