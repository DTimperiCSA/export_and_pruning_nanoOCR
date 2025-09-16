from pathlib import Path
from Levenshtein import distance as lev_distance
from jiwer import wer, cer
from difflib import SequenceMatcher

from paths import *

def similarity(a, b):
    """Calcola la percentuale di similarità tra due testi"""
    return SequenceMatcher(None, a, b).ratio() * 100

def read_file(path):
    """Legge un file di testo e restituisce il contenuto"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Percorso alla cartella contenente i file
folder_path = Path(OCR_TEST_PATH)  # modifica con la tua cartella

# Lista tutti i file .txt nella cartella
txt_files = list(folder_path.glob("*.txt"))

# Filtra i file principali (senza _fp16_)
main_files = [f for f in txt_files if "_fp16_" not in f.name]

# Print header
for main_file in main_files:
    base_name = main_file.stem
    text_main = read_file(main_file)
    
    # Define files to compare
    noresize_file = folder_path / f"{base_name}_fp16_noresize.txt"
    resized_file = folder_path / f"{base_name}_fp16_resized.txt"
    
    if noresize_file.exists() and resized_file.exists():
        text_noresize = read_file(noresize_file)
        text_resized = read_file(resized_file)
        
        # Metrics vs main file
        sim_noresize = similarity(text_main, text_noresize)
        lev_noresize = lev_distance(text_main, text_noresize)
        lev_norm_noresize = lev_noresize / max(len(text_main), len(text_noresize))
        cer_noresize = cer(text_main, text_noresize)
        wer_noresize = wer(text_main, text_noresize)
        
        sim_resized = similarity(text_main, text_resized)
        lev_resized = lev_distance(text_main, text_resized)
        lev_norm_resized = lev_resized / max(len(text_main), len(text_resized))
        cer_resized = cer(text_main, text_resized)
        wer_resized = wer(text_main, text_resized)
        
        # Print all metrics on one line
        print("\n")
        print(main_file)
        print(f"fp16_noresize vs fp16_resized")
        print(f"Similarity: {sim_noresize:.2f}% vs {sim_resized:.2f}%")
        print(f"Levenshtein: {lev_noresize} vs {lev_resized}")
        print(f"Normalized Levenshtein: {lev_norm_noresize:.3f} vs {lev_norm_resized:.3f}")
        print(f"CER: {cer_noresize:.3f} vs {cer_resized:.3f}")
        print(f"WER: {wer_noresize:.3f} vs {wer_resized:.3f}")



#10, 17, 18