import re
import string
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from paths import *

# ─────────────────────────────────────────────
# Build a global stopword set (all languages in NLTK)
# ─────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    STOPWORDS = set()
    for lang in stopwords.fileids():
        STOPWORDS.update(stopwords.words(lang))
except Exception:
    STOPWORDS = {
        "and",
        "or",
        "but",
        "if",
        "while",
        "although",
        "though",
        "because",
        "e",
        "o",
        "ma",
        "se",
        "aunque",
        "pero",
        "y",
        "ou",
        "mas",
        "si",
    }


# ─────────────────────────────────────────────
def tokenize(text):
    """Tokenize, remove punctuation, numbers, stopwords (any language)."""
    translator = str.maketrans("", "", string.punctuation)
    text = text.lower().translate(translator)

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def generate_ngrams(tokens, n):
    """Generate n-grams from list of tokens."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ─────────────────────────────────────────────
def process_txt_files(root_path, ngram_range=(1, 2, 3, 4)):
    root_path = Path(root_path)

    global_bows = {n: Counter() for n in ngram_range}
    folder_bows = defaultdict(lambda: {n: Counter() for n in ngram_range})

    for txt_file in root_path.rglob("*.txt"):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Could not read {txt_file}: {e}")
            continue

        tokens = tokenize(text)

        for n in ngram_range:
            ngrams = generate_ngrams(tokens, n)
            folder_bows[txt_file.parent][n].update(ngrams)
            global_bows[n].update(ngrams)

    return global_bows, folder_bows


# ─────────────────────────────────────────────
def save_bows(global_bows, folder_bows, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Save global bag of words
    for n, bow in global_bows.items():
        global_file = output_folder / f"global_{n}gram.txt"
        with global_file.open("w", encoding="utf-8") as f:
            for word, count in bow.most_common():
                f.write(f"{word} {count}\n")

    # Save folder/subfolder bags of words
    for folder, bows in folder_bows.items():
        safe_name = "_".join(folder.parts[-2:])  # avoid collisions
        for n, bow in bows.items():
            file_path = output_folder / f"{safe_name}_{n}gram.txt"
            with file_path.open("w", encoding="utf-8") as f:
                for word, count in bow.most_common():
                    f.write(f"{word} {count}\n")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    root_directory = TO_TEST_OCR
    output_directory = (
        Path("/home/lucapolenta/Desktop/export_and_pruning_nanoOCR/") / "bag_of_words"
    )

    global_bows, folder_bows = process_txt_files(
        root_directory, ngram_range=(1, 2, 3, 4)
    )
    save_bows(global_bows, folder_bows, output_directory)

    print(f"Bags of words (1–4 grams) saved in: {output_directory}")
