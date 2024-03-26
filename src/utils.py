from pathlib import Path
from typing import Optional

import tiktoken
import fitz


def num_tokens(string: Optional[str]) -> int:
    if not string:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def load_doc(
    file: str | Path | fitz.Document,
) -> fitz.Document:
    if isinstance(file, str):
        pdf = fitz.open(Path(file))
    elif isinstance(file, Path):
        pdf = fitz.open(file)
    elif isinstance(file, fitz.Document):
        pdf = fitz.open()
        pdf.insert_pdf(file)
    else:
        raise TypeError(f"Invalid type for file: {type(file)}")
    return pdf


def extract_pages(
    input_pdf: Path, output_pdf: Path, start_page: int, end_page: int
) -> None:
    """
    Utility function for generating evals.
    """
    doc = fitz.open(input_pdf)

    new_doc = fitz.open()

    for page_num in range(start_page, end_page + 1):
        page = doc.load_page(page_num)
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    new_doc.save(output_pdf)

    doc.close()
    new_doc.close()
