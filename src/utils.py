from pathlib import Path

import fitz


def load_doc(
    file: str | Path | fitz.Document,
) -> fitz.Document:
    if isinstance(file, str) or isinstance(file, Path):
        pdf = fitz.open(file)
    elif isinstance(file, fitz.Document):
        pdf = fitz.open()
        pdf.insert_pdf(file)
    else:
        raise TypeError(f"Invalid type for file: {type(file)}")
    return pdf
