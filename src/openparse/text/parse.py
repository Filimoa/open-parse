from typing import List, Literal

from openparse.pdf import Pdf
from openparse.schemas import TextElement

from . import pdfminer, pymupdf


def ingest(
    doc: Pdf, parsing_method: Literal["pdfminer", "pymupdf"] = "pdfminer"
) -> List[TextElement]:
    """
    Default to pdfminer-based implementation.

    Optional use the PyMuPDF-based implementation which has identical behaviour but supports OCR. Important - see their licensing to view the terms of use:
    https://mupdf.com/licensing/index.html
    """
    if parsing_method == "pdfminer":
        return pdfminer.ingest(doc)
    elif parsing_method == "pymupdf":
        return pymupdf.ingest(doc)
    else:
        raise ValueError(f"Unsupported parsing_method: {parsing_method}")
