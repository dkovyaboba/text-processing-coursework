"""
Точка входа программы.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cleaner import clean_text
from utils import read_file, count_frequencies, save_results, count_sentences
from analytics import extract_ngrams, analyze_sentiment, classify_style, calculate_readability

def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ текстовых данных")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Входной файл")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Выходной файл")
    parser.add_argument("--language", "-l", default="russian", choices=["russian", "english"])
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Файл не найден: {args.input}", file=sys.stderr)
        sys.exit(1)

    raw_text = read_file(args.input)
    tokens = clean_text(raw_text, language=args.language)
    
    # Сохраняем базовые частоты
    frequencies = count_frequencies(tokens)
    save_results(frequencies, args.output)
    
    # Аналитика
    bigrams = extract_ngrams(tokens, n=2)
    bigram_freqs = count_frequencies([" ".join(bg) for bg in bigrams])
    sentiment = analyze_sentiment(tokens)
    style_scores = classify_style(tokens)
    predicted_style = max(style_scores, key=style_scores.get)
    sentences_count = count_sentences(raw_text)
    readability = calculate_readability(tokens, sentences_count)
    
    print("\n" + "="*40)
    print("📊 АНАЛИТИКА ТЕКСТА")
    print("="*40)
    print(f"Топ-5 биграмм: {bigram_freqs[:5]}")
    print(f"Тональность: Позитив={sentiment['positive']}, Негатив={sentiment['negative']}, Оценка={sentiment['score']}")
    print(f"🎭 Стиль: {predicted_style} (совпадений: {style_scores[predicted_style]})")
    print(f"Сложность (Индекс Флеша): {readability}")
    print(f"[INFO] Результат сохранён в {args.output}")

if __name__ == "__main__":
    main()
    