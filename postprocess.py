from pathlib import Path

def remove_before_headline(input_file: str, output_file: str = None):
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"{input_file} does not exist")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the first line containing "assistant"
    for i, line in enumerate(lines):
        if "assistant" in line:
            # Keep everything after this line
            cleaned_lines = lines[i+1:]
            break
    else:
        # If "assistant" not found, return empty
        cleaned_lines = []

    # Write back to the same file or a new file
    output_path = Path(output_file) if output_file else input_path
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)