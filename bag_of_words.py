from pathlib import Path
from collections import Counter, defaultdict
import string
from paths import *

def tokenize(text):
    """Simple tokenizer: lowercase, remove punctuation, split by whitespace."""
    translator = str.maketrans('', '', string.punctuation)
    return text.lower().translate(translator).split()

def process_txt_files(root_path):
    root_path = Path(root_path)
    
    global_bow = Counter()
    folder_bows = defaultdict(Counter)
    
    for txt_file in root_path.rglob("*.txt"):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Could not read {txt_file}: {e}")
            continue
        
        words = tokenize(text)
        folder_bows[txt_file.parent].update(words)
        global_bow.update(words)
    
    return global_bow, folder_bows

def save_bows(global_bow, folder_bows, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Save global bag of words
    global_file = output_folder / "global.txt"
    with global_file.open("w", encoding="utf-8") as f:
        for word, count in global_bow.most_common():
            f.write(f"{word} {count}\n")
    
    # Save folder/subfolder bags of words
    for folder, bow in folder_bows.items():
        # Make folder name safe for filename
        safe_name = "_".join(folder.parts[-2:])  # last two parts to avoid collisions
        file_path = output_folder / f"{safe_name}.txt"
        with file_path.open("w", encoding="utf-8") as f:
            for word, count in bow.most_common():
                f.write(f"{word} {count}\n")

if __name__ == "__main__":
    root_directory = BELGRADO_OCR
    output_directory = Path(r"C:\Belgrado") / "bag_of_words"
    
    global_bow, folder_bows = process_txt_files(root_directory)
    save_bows(global_bow, folder_bows, output_directory)
    
    print(f"Bags of words saved in: {output_directory}")
