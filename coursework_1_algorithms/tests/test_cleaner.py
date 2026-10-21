"""
Юнит-тесты для модуля очистки текста.
Запуск: pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cleaner import (
    remove_html_tags,
    normalize_whitespace,
    remove_punctuation,
    tokenize,
    clean_text,
)


def test_remove_html_tags():
    assert remove_html_tags("<p>Привет</p>") == "Привет"
    assert remove_html_tags("<div><b>Текст</b></div>") == "Текст"


def test_html_not_parsed_for_plain_text():
    """Текст с '<' и '>' без HTML-тегов не должен ломаться."""
    text = "a < 5 and b > 10"
    result = remove_html_tags(text)
    assert result == text  # BeautifulSoup не вызывался


def test_normalize_whitespace():
    assert normalize_whitespace("a   b\n\nc") == "a b c"


def test_remove_punctuation_does_not_glue():
    """Пунктуация заменяется на пробел, слова не склеиваются."""
    result = remove_punctuation("веб-разработка")
    assert "веб разработка" == result


def test_yo_letter_preserved():
    """Буква ё не должна теряться при удалении пунктуации."""
    result = remove_punctuation("ёлка, ещё!")
    assert "ёлка" in result
    assert "ещё" in result


def test_tokenize():
    assert tokenize("a b c") == ["a", "b", "c"]


def test_none_input():
    """None не должен вызывать исключение."""
    assert clean_text(None) == []


def test_clean_text_russian():
    raw = "<p>Привет, мир! Это тестовый текст.</p>"
    result = clean_text(raw, language="russian")
    # Проверяем, что стоп-слово "это" удалено
    assert "это" not in result
    # Проверяем, что список не пустой и содержит результаты
    assert len(result) > 0
    # Проверяем, что слово "привет" осталось (в лемматизированном виде)
    assert any("привет" in word for word in result)

def test_clean_text_english():
    raw = "<p>Hello, world! This is a test.</p>"
    result = clean_text(raw, language="english")
    # Проверяем, что стоп-слово "this" удалено
    assert "this" not in result
    # Проверяем, что список не пустой
    assert len(result) > 0
    # Проверяем, что слово "hello" осталось
    assert "hello" in result
