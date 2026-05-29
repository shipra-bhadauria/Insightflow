import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdfplumber


def load_pdf(file_path: str) -> dict:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages_text = []
    full_text  = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        max_pages = min(len(pdf.pages), 15)
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text()
            if text:
                text = text.strip()
                pages_text.append({
                    "page":  i + 1,
                    "text":  text,
                    "chars": len(text),
                })
                full_text.append(text)

    combined_text = "\n\n".join(full_text)

    return {
        "file_path":    file_path,
        "total_pages":  total_pages,
        "pages_loaded": len(pages_text),
        "total_chars":  len(combined_text),
        "pages":        pages_text,
        "full_text":    combined_text,
        "preview":      combined_text[:500],
    }


def extract_tables_from_pdf(file_path: str) -> dict:

    tables = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages[:20]):
            page_tables = page.extract_tables()
            for j, table in enumerate(page_tables):
                if table:
                    tables.append({
                        "page":    i + 1,
                        "table":   j + 1,
                        "headers": table[0] if table else [],
                        "rows":    table[1:] if len(table) > 1 else [],
                        "n_rows":  len(table) - 1,
                    })

    return {
        "file_path":   file_path,
        "total_tables": len(tables),
        "tables":      tables,
    }


if __name__ == "__main__":
    import sys

    # create a simple test PDF if no file provided
    file_path = sys.argv[1] if len(sys.argv) > 1 else None

    if not file_path:
        print("Usage: python tools/pdf_loader.py <path_to_pdf>")
        print("\nNo PDF provided — creating a test PDF...")

        try:
            from reportlab.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet

            test_path = "outputs/test_doc.pdf"
            os.makedirs("outputs", exist_ok=True)

            doc    = SimpleDocTemplate(test_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story  = [
                Paragraph("InsightFlow Test Document", styles["Title"]),
                Paragraph("Asia leads revenue at £1.94M average per order.", styles["Normal"]),
                Paragraph("Sub-Saharan Africa has the lowest at £1.10M.", styles["Normal"]),
                Paragraph("Total revenue declined 9.17% over the period.", styles["Normal"]),
            ]
            doc.build(story)
            file_path = test_path
            print(f"Test PDF created at {test_path}\n")
        except Exception as e:
            print(f"Could not create test PDF: {e}")
            sys.exit(1)

    result = load_pdf(file_path)

    print("=== PDF Loader Output ===\n")
    print(f"File:         {result['file_path']}")
    print(f"Total pages:  {result['total_pages']}")
    print(f"Pages loaded: {result['pages_loaded']}")
    print(f"Total chars:  {result['total_chars']}")
    print(f"\nPreview:\n{result['preview']}")

    tables = extract_tables_from_pdf(file_path)
    print(f"\nTables found: {tables['total_tables']}")