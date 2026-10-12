"""Тесты словаря кухонь и текстов-описаний для семантического поиска."""

from ai_course_agents.search.cuisine import cuisine_words
from ai_course_agents.search.documents import describe


def test_cuisine_tags_become_russian_dishes():
    # Несколько тегов через «;», регистр и пробелы не мешают
    words = cuisine_words(" Russian; pizza ")
    assert "пельмени" in words and "пицца" in words


def test_unknown_cuisine_tag_is_kept_readable():
    # Неизвестный тег не теряется: подчёркивания → пробелы, модель эмбеддингов поймёт и так
    assert cuisine_words("bubble_tea") == "bubble tea"
    assert cuisine_words(None) is None


def test_describe_uses_facts_and_marks_guess():
    place = {"kind": "restaurant", "name": "Марьяж", "brand": None, "cuisine": "russian"}
    text = describe(place, guess="домашняя еда")
    assert text.startswith("Ресторан «Марьяж»")
    assert "кухня: русская кухня: пельмени" in text
    assert "предположительно: домашняя еда" in text     # догадка LLM помечена как догадка


def test_describe_skips_brand_equal_to_name():
    place = {"kind": "supermarket", "name": "Магнит", "brand": "Магнит", "cuisine": None}
    assert describe(place) == "Супермаркет «Магнит»"
