import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytesseract
from PIL import Image
import pandas as pd
import re


def load_image(file_path: str) -> dict:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found: {file_path}")

    # open image
    img  = Image.open(file_path)
    width, height = img.size
    mode = img.mode

    # extract text using OCR
    raw_text = pytesseract.image_to_string(img)
    clean_text = raw_text.strip()

    # get word count
    words = [w for w in clean_text.split() if w.strip()]

    return {
        "file_path":  file_path,
        "width":      width,
        "height":     height,
        "mode":       mode,
        "raw_text":   clean_text,
        "word_count": len(words),
        "preview":    clean_text[:500],
        "source_type": "image",
    }


def image_to_dataframe(file_path: str) -> pd.DataFrame:
    img    = Image.open(file_path)
    data   = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT
    )

    # extract rows of text with confidence > 60
    rows   = []
    current_row = []
    last_line   = -1

    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i])
        line = data["line_num"][i]

        if conf > 60 and text:
            if line != last_line and current_row:
                rows.append(" ".join(current_row))
                current_row = []
            current_row.append(text)
            last_line = line

    if current_row:
        rows.append(" ".join(current_row))

    # try to detect if rows look like tabular data
    # split by multiple spaces
    table_rows = []
    for row in rows:
        cols = re.split(r"\s{2,}", row)
        if len(cols) > 1:
            table_rows.append(cols)

    if table_rows and len(table_rows) > 1:
        # try to use first row as headers
        try:
            headers = table_rows[0]
            data_rows = table_rows[1:]
            # pad shorter rows
            max_cols = len(headers)
            data_rows = [r + [""] * (max_cols - len(r)) for r in data_rows]
            df = pd.DataFrame(data_rows, columns=headers)
            return df
        except Exception:
            pass

    # fallback — return single column df with text rows
    return pd.DataFrame({"text": rows})


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1] if len(sys.argv) > 1 else None

    if not file_path:
        # test with one of our chart images
        chart_files = [f for f in os.listdir("outputs") if f.endswith(".png")]
        if chart_files:
            file_path = os.path.join("outputs", chart_files[0])
            print(f"Testing with: {file_path}\n")
        else:
            print("No image found. Usage: python tools/image_loader.py <image_path>")
            sys.exit(1)

    result = load_image(file_path)

    print("=== Image Loader Output ===\n")
    print(f"File:       {result['file_path']}")
    print(f"Size:       {result['width']} × {result['height']}")
    print(f"Mode:       {result['mode']}")
    print(f"Words:      {result['word_count']}")
    print(f"\nExtracted text preview:\n{result['preview'][:300] if result['preview'] else 'none'}")