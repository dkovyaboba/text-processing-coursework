"""
Модуль очистки и нормализации текстовых данных.
Содержит алгоритмы предварительной обработки текста:
удаление HTML, пунктуации, стоп-слов, лемматизация.
"""
import re
from typing import List, Set

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data() -> None:
    """Загружает необходимые ресурсы NLTK при первом запуске."""
    for resource in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def remove_html_tags(text: str) -> str:
    """
    Удаляет HTML-теги из строки.
    
    Алгоритм: использует BeautifulSoup для парсинга,
    возвращает только текстовое содержимое.
    Сложность: O(n), где n — длина текста.
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def normalize_whitespace(text: str) -> str:
    """
    Заменяет множественные пробелы и переносы на одиночный пробел.
    
    Алгоритм: регулярное выражение \s+ → " ".
    Сложность: O(n).
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str) -> str:
    """
    Удаляет знаки пунктуации и цифры.
    
    Алгоритм: регулярное выражение [^a-zA-Zа-яА-Я\s] → "".
    Сложность: O(n).
    """
    return re.sub(r"[^a-zA-Zа-яА-Я\s]", "", text)


def tokenize(text: str) -> List[str]:
    """
    Разбивает текст на токены (слова) по пробелам.
    
    Сложность: O(n).
    """
    return text.split()


def remove_stopwords(tokens: List[str], language: str = "russian") -> List[str]:
    """
    Удаляет стоп-слова из списка токенов.
    
    Алгоритм: проверка каждого токена на вхождение в множество стоп-слов.
    Сложность: O(m), где m — количество токенов (проверка в set — O(1)).
    """
    ensure_nltk_data()
    stop_words: Set[str] = set(stopwords.words(language))
    return [token for token in tokens if token.lower() not in stop_words]


def lemmatize(tokens: List[str]) -> List[str]:
    """
    Приводит слова к начальной форме (лемматизация).
    
    Алгоритм: WordNetLemmatizer для каждого токена.
    Сложность: O(m * k), где m — число токенов, k — средняя длина слова.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]


def clean_text(raw_text: str, language: str = "russian") -> List[str]:
    """
    Полный конвейер очистки текста.
    
    Этапы:
    1. Удаление HTML-тегов
    2. Нормализация пробелов
    3. Удаление пунктуации
    4. Токенизация
    5. Удаление стоп-слов
    6. Лемматизация
    
    Возвращает список очищенных лемм.
    """
    text = remove_html_tags(raw_text)
    text = normalize_whitespace(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, language=language)
    tokens = lemmatize(tokens)
    return tokens
