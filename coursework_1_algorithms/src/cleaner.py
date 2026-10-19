"""
Модуль очистки и нормализации текстовых данных.
Поддерживает русский и английский языки.
Лемматизация: pymorphy3 (русский), WordNetLemmatizer (английский).
Все тяжёлые ресурсы инициализируются один раз на уровне модуля.
"""
import re
from functools import lru_cache
from typing import List, Optional, Set

import pymorphy3
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Инициализация ресурсов (один раз при импорте модуля) ---

# pymorphy3: морфологический анализатор для русского языка
_MORPH_RU = pymorphy3.MorphAnalyzer()

# WordNetLemmatizer: только для английского
_LEMMATIZER_EN = WordNetLemmatizer()

# Стоп-слова: загружаются один раз
_STOPWORDS_CACHE: dict[str, Set[str]] = {}


def _get_stopwords(language: str) -> Set[str]:
    """Возвращает кэшированное множество стоп-слов для языка."""
    if language not in _STOPWORDS_CACHE:
        _STOPWORDS_CACHE[language] = set(stopwords.words(language))
    return _STOPWORDS_CACHE[language]


# --- Безопасное удаление HTML ---

# Регулярное выражение для «похоже на HTML»: ищем парные теги
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")


def remove_html_tags(text: Optional[str]) -> str:
    """
    Удаляет HTML-теги, только если текст действительно похож на HTML.
    Для обычного текста с символами < и > (например, "a < 5") парсер не вызывается.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Если нет ни одного HTML-подобного тега — не тратим ресурсы на BeautifulSoup
    if not _HTML_TAG_RE.search(text):
        return text

    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


# --- Нормализация и очистка ---

def to_lowercase(text: str) -> str:
    """Приводит весь текст к нижнему регистру один раз в начале конвейера."""
    return text.lower()


def normalize_whitespace(text: str) -> str:
    """
    Схлопывает любые последовательности пробельных символов в один пробел.
    Оставлено для явности, хотя split() далее сделает то же самое.
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str) -> str:
    """
    Заменяет знаки пунктуации и цифры на одиночный пробел (не удаляет!).
    Это предотвращает склеивание слов: "веб-разработка" → "веб разработка",
    "конец.Начало" → "конец Начало".
    Буквы «ё» и «Ё» включены явно: [ёЁ].
    """
    # Всё, что НЕ буква (латиница, кириллица с ё) и НЕ пробел → пробел
    return re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]", " ", text)


def tokenize(text: str) -> List[str]:
    """
    Разбивает текст на токены по пробелам.
    split() без аргументов сам схлопывает множественные пробелы.
    """
    return text.split()


def remove_stopwords(tokens: List[str], language: str = "russian") -> List[str]:
    """
    Удаляет стоп-слова. Токены уже в нижнем регистре (приведение выполнено ранее).
    Множество стоп-слов кэшировано.
    """
    stop_words = _get_stopwords(language)
    return [token for token in tokens if token not in stop_words]


def lemmatize(tokens: List[str], language: str = "russian") -> List[str]:
    """
    Лемматизация с выбором инструмента по языку.
    Для русского — pymorphy3 (корректно приводит к начальной форме).
    Для английского — WordNetLemmatizer.
    """
    if language == "russian":
        return [_MORPH_RU.parse(token)[0].normal_form for token in tokens]
    elif language == "english":
        return [_LEMMATIZER_EN.lemmatize(token) for token in tokens]
    else:
        raise ValueError(f"Unsupported language: {language}")


def clean_text(raw_text: Optional[str], language: str = "russian") -> List[str]:
    """
    Полный конвейер очистки текста.
    
    Порядок операций (оптимизирован):
    1. Валидация (None / не строка → пустой список)
    2. Удаление HTML (только если текст похож на HTML)
    3. Приведение к нижнему регистру (один раз)
    4. Замена пунктуации на пробелы (не удаление!)
    5. Токенизация (split сам схлопывает пробелы)
    6. Удаление стоп-слов
    7. Лемматизация (pymorphy3 для русского)
    """
    # 1. Валидация
    if raw_text is None or not isinstance(raw_text, str):
        return []

    # 2. HTML (безопасно)
    text = remove_html_tags(raw_text)

    # 3. Нижний регистр (один раз)
    text = to_lowercase(text)

    # 4. Пунктуация → пробелы (сохраняет границы слов, включает ё/Ё)
    text = remove_punctuation(text)

    # 5. Токенизация (split сам убирает лишние пробелы)
    tokens = tokenize(text)

    # 6. Стоп-слова
    tokens = remove_stopwords(tokens, language=language)

    # 7. Лемматизация
    tokens = lemmatize(tokens, language=language)

    return tokens
