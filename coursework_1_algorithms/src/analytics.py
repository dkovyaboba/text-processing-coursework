"""
Модуль аналитики: поиск N-грамм, анализ тональности, классификация стиля.
"""
from typing import List, Tuple, Dict
from collections import Counter

# Словари тональности
POSITIVE_WORDS = {"хороший", "отличный", "прекрасный", "супер", "нравиться", "любить", "успех", "быстрый", "удобный", "качественный", "рекомендую"}
NEGATIVE_WORDS = {"плохой", "ужасный", "отвратительный", "медленный", "сложный", "проблема", "ошибка", "провал", "ненавидеть", "скучный", "разочарование"}

# Маркеры стилей
STYLE_MARKERS = {
    "научный": {"исследование", "метод", "анализ", "результат", "данные", "теория", "эксперимент", "система"},
    "новостной": {"сообщил", "заявил", "произошел", "сегодня", "полиция", "власти", "событие", "регион"},
    "разговорный": {"привет", "спасибо", "круто", "отстой", "ребята", "вообще", "просто", "класс", "бомба"}
}

def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Извлекает N-граммы из списка токенов.
    Алгоритм: скользящее окно (sliding window) размером n.
    Сложность: O(m * n), где m — число токенов.
    """
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def analyze_sentiment(tokens: List[str]) -> Dict[str, float]:
    """Анализирует тональность текста на основе словарей."""
    pos_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    total = len(tokens)
    if total == 0:
        return {"positive": 0, "negative": 0, "score": 0.0}
    score = (pos_count - neg_count) / total
    return {"positive": pos_count, "negative": neg_count, "score": round(score, 3)}

def classify_style(tokens: List[str]) -> Dict[str, int]:
    """Определяет стиль текста по наличию маркерных слов."""
    scores = {style: 0 for style in STYLE_MARKERS}
    for token in tokens:
        for style, markers in STYLE_MARKERS.items():
            if token in markers:
                scores[style] += 1
    return scores

def calculate_readability(tokens: List[str], sentences_count: int) -> float:
    """Простейшая оценка сложности (индекс Флеша для русского)."""
    if sentences_count == 0 or len(tokens) == 0:
        return 0.0
    avg_word_length = sum(len(word) for word in tokens) / len(tokens)
    avg_sentence_length = len(tokens) / sentences_count
    score = 206.835 - (1.3 * avg_sentence_length) - (60.1 * avg_word_length)
    return round(score, 2)

