import sys
import os
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

def load_image(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found: {file_path}")

    img = Image.open(file_path)
    width, height = img.size
    mode = img.mode

    # VLM Analysis (gpt-4o vision)
    vlm_description = ""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = file_path.split(".")[-1].lower()
        media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. "
                            "If it contains data, charts, tables, or text — "
                            "extract and explain all key information. "
                            "Be specific about numbers, labels, trends, and insights."
                        ),
                    },
                ],
            }],
            max_tokens=1000,
        )
        vlm_description = response.choices[0].message.content
    except Exception as e:
        vlm_description = f"[VLM analysis failed: {e}]"

    return {
        "file_path":       file_path,
        "width":           width,
        "height":          height,
        "mode":            mode,
        "raw_text":        vlm_description,
        "word_count":      len(vlm_description.split()),
        "preview":         vlm_description[:500],
        "source_type":     "image",
        "vlm_description": vlm_description,
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