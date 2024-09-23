import datetime as dt
import io
import mimetypes
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage
from PIL import Image
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter

from openparse.schemas import Bbox, Node


class _BboxWithColor(BaseModel):
    color: Tuple[float, float, float]
    bbox: Bbox
    annotation_text: Optional[Any] = None


def _random_color() -> Tuple[float, float, float]:
    return (
        random.randint(0, 255) / 256,
        random.randint(0, 255) / 256,
        random.randint(0, 255) / 256,
    )


def _prepare_bboxes_for_drawing(
    bboxes: Union[List[Bbox], List[List[Bbox]]], annotations: Optional[List[str]] = None
) -> List[_BboxWithColor]:
    res = []
    assert (
        len(bboxes) == len(annotations) if annotations else True
    ), "Number of annotations must match the number of bboxes."

    for element in bboxes:
        color = _random_color()
        text = annotations.pop(0) if annotations else None
        if isinstance(element, Bbox):
            res.append(
                _BboxWithColor(
                    color=color,
                    bbox=element,
                    annotation_text=text,
                )
            )
        elif isinstance(element, list):
            sorted_bboxes = sorted(element, key=lambda x: x.page)
            for bbox in sorted_bboxes:
                res.append(
                    _BboxWithColor(
                        color=color,
                        bbox=bbox,
                        annotation_text=text,
                    )
                )

                text = "continued ..."
    return res


def file_metadata(file_path: Union[str, Path]) -> Dict:
    """Get some handy metadate from filesystem.

    Args:
        file_path: str: file path in str
    """
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_type": mimetypes.guess_type(file_path)[0],
        "file_size": os.path.getsize(file_path),
        "creation_date": dt.datetime.fromtimestamp(
            Path(file_path).stat().st_ctime
        ).strftime("%Y-%m-%d"),
        "last_modified_date": dt.datetime.fromtimestamp(
            Path(file_path).stat().st_mtime
        ).strftime("%Y-%m-%d"),
        "last_accessed_date": dt.datetime.fromtimestamp(
            Path(file_path).stat().st_atime
        ).strftime("%Y-%m-%d"),
    }


class Pdf:
    """
    Simple utility class for working with PDF files. This class wraps the PdfReader and PdfWriter classes from pypdf.
    """

    def __init__(self, file: Union[str, Path, PdfReader]):
        self.file_path = None
        self.file_metadata = {}
        if isinstance(file, (str, Path)):
            self.file_path = str(file)
            self.file_metadata = file_metadata(file)

        self.reader = PdfReader(file) if isinstance(file, (str, Path)) else file
        self.writer = PdfWriter()
        for page in self.reader.pages:
            self.writer.add_page(page)

        self.num_pages = len(self.reader.pages)

    def extract_layout_pages(self) -> Iterator[LTPage]:
        """
        Yields layout objects for each page in the PDF using pdfminer.six.
        """
        assert (
            self.file_path is not None
        ), "PDF file path is required for this method for now."

        yield from extract_pages(self.file_path)

    def save(self, output_pdf: Union[str, Path]) -> None:
        """
        Saves the content from the PdfWriter to a new PDF file.
        """
        with open(str(output_pdf), "wb") as out_pdf:
            self.writer.write(out_pdf)

    def extract_pages(self, start_page: int, end_page: int) -> None:
        """
        Extracts a range of pages from the PDF and adds them to the PdfWriter.
        """
        for page_num in range(start_page - 1, end_page):
            self.writer.add_page(self.reader.pages[page_num])

    def to_pymupdf_doc(self):
        """
        Transforms the PDF into a PyMuPDF (fitz) document.
        If modifications have been made using PdfWriter, it saves to a temporary file first.
        This function dynamically imports PyMuPDF (fitz), requiring it only if this method is called.
        """
        try:
            import fitz  # type: ignore
        except ImportError as err:
            raise ImportError(
                "PyMuPDF (fitz) is not installed. This method requires PyMuPDF."
            ) from err

        if not self.writer.pages:
            return fitz.open(self.file_path)

        byte_stream = io.BytesIO()
        self.writer.write(byte_stream)
        return fitz.open(None, byte_stream)

    def _draw_bboxes(
        self,
        bboxes_with_color: List[_BboxWithColor],
        coordinates: Literal["top-left", "bottom-left"],
    ):
        try:
            import fitz
        except ImportError as err:
            raise ImportError(
                "PyMuPDF (fitz) is not installed. This method requires PyMuPDF."
            ) from err

        pdf = self.to_pymupdf_doc()

        for page in pdf:
            page.wrap_contents()

            for bbox_with_color in bboxes_with_color:
                bbox = bbox_with_color.bbox
                if bbox.page != page.number:
                    continue
                if coordinates == "bottom-left":
                    bbox = self._flip_coordinates(bbox)
                rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                page.draw_rect(
                    rect, bbox_with_color.color
                )  # Use the color associated with this bbox

                if bbox_with_color.annotation_text:
                    page.insert_text(
                        rect.top_left,
                        str(bbox_with_color.annotation_text),
                        fontsize=12,
                    )
        return pdf

    def display_with_bboxes(
        self,
        nodes: List[Node],
        page_nums: Optional[List[int]] = None,
        annotations: Optional[List[str]] = None,
    ):
        """
        Display a single page of a PDF file using IPython.
        Optionally, display a piece of text on top of the bounding box.
        """
        try:
            from IPython.display import Image, display  # type: ignore
        except ImportError as err:
            raise ImportError(
                "IPython is required to display PDFs. Please install it with `pip install ipython`."
            ) from err
        assert nodes, "At least one node is required."

        bboxes = [node.bbox for node in nodes]
        flattened_bboxes = _prepare_bboxes_for_drawing(bboxes, annotations)
        marked_up_doc = self._draw_bboxes(flattened_bboxes, nodes[0].coordinate_system)
        if not page_nums:
            page_nums = list(range(marked_up_doc.page_count))
        for page_num in page_nums:
            page = marked_up_doc[page_num]
            img_data = page.get_pixmap().tobytes("png")
            display(Image(data=img_data))

    def export_with_bboxes(
        self,
        nodes: List[Node],
        output_pdf: Union[str, Path],
        annotations: Optional[List[str]] = None,
    ) -> None:
        assert nodes, "At least one node is required."

        bboxes = [node.bbox for node in nodes]
        flattened_bboxes = _prepare_bboxes_for_drawing(bboxes, annotations)
        marked_up_doc = self._draw_bboxes(flattened_bboxes, nodes[0].coordinate_system)
        marked_up_doc.save(str(output_pdf))

    def _flip_coordinates(self, bbox: Bbox) -> Bbox:
        fy0 = bbox.page_height - bbox.y1
        fy1 = bbox.page_height - bbox.y0
        return Bbox(
            page=bbox.page,
            page_height=bbox.page_height,
            page_width=bbox.page_width,
            x0=bbox.x0,
            y0=fy0,
            x1=bbox.x1,
            y1=fy1,
        )

    def to_imgs(self, page_numbers: Optional[List[int]] = None) -> List[Image.Image]:
        doc = self.to_pymupdf_doc()
        images = []

        if not doc.is_pdf:
            raise ValueError("The document is not in PDF format.")
        if doc.needs_pass:
            raise ValueError("The PDF document is password protected.")

        if page_numbers is None:
            page_numbers = list(range(doc.page_count))

        for n in page_numbers:
            page = doc[n]
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(image)

        return images
