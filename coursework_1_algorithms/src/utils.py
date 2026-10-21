"""
Вспомогательные функции: чтение файла, подсчёт частот, сохранение.
"""
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

def read_file(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def count_frequencies(tokens: List[str]) -> List[Tuple[str, int]]:
    counter = Counter(tokens)
    return counter.most_common()

def save_results(results: List[Tuple[str, int]], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for word, freq in results:
            f.write(f"{word}\t{freq}\n")

def count_sentences(text: str) -> int:
    """Считает количество предложений по знакам конца предложения."""
    sentences = re.findall(r'[.!?]+', text)
    return max(1, len(sentences))
