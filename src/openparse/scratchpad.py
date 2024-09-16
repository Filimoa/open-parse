import asyncio
import datetime as dt
import time
import uuid
from collections import namedtuple
from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

import google.generativeai as gemini_sdk
from PIL import Image, ImageChops, ImageFilter
from pydantic import BaseModel, Field

from openparse import consts
from openparse.pdf import Pdf
from openparse.schemas import Bbox

## If you come across a table, please use HTML <table> tags to represent the table and embed within the Markdown.

PARSE_DOC_PROMPT = """\
OCR the text on the following page to Markdown. Respect line breaks.
"""


class RecitationError(Exception):
    """https://issuetracker.google.com/issues/331677495"""

    pass


class GeminiBatchArgsDict(TypedDict):
    model: str = Literal["gemini-1.5-flash"] | Literal["gemini-1.5-pro"] | str
    api_key: str
    max_batch_size: int = 150_000


class _LlmExtractedPage(BaseModel):
    page_num: int
    bbox: Bbox  # cropped around the page
    text: str
    tokens: int


class LlmParsedDocument(BaseModel):
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID of the node.",
        exclude=True,
    )
    pages: List[_LlmExtractedPage]
    filename: str
    num_pages: int
    coordinate_system: Literal["top-left", "bottom-left"] = "bottom-left"
    table_parsing_kwargs: Optional[dict] = None
    last_modified_date: Optional[dt.date] = None
    last_accessed_date: Optional[dt.date] = None
    creation_date: Optional[dt.date] = None
    file_size: Optional[int] = None


def _get_content_bounding_box(image: Image.Image) -> tuple[int, int, int, int]:
    grayscale = image.convert("L")
    inverted = ImageChops.invert(grayscale)
    blurred = inverted.filter(ImageFilter.GaussianBlur(radius=2))
    binary = blurred.point(lambda x: 0 if x < 50 else 255, "1")
    bbox = binary.getbbox()

    padding = 10
    width, height = image.size
    assert bbox is not None
    bbox = (
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(width, bbox[2] + padding),
        min(height, bbox[3] + padding),
    )

    return bbox


def _get_content_bounding_boxes(pdf_obj: Pdf) -> list[Bbox]:
    images = pdf_obj.to_imgs()
    bounding_boxes = []

    for i, image in enumerate(images):
        page_num = i
        bbox = _get_content_bounding_box(image)

        if bbox:
            x0, y0, x1, y1 = bbox

            bbox_model = Bbox(
                page=page_num,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                page_width=image.width,
                page_height=image.height,
            )
            bounding_boxes.append(bbox_model)

    return bounding_boxes


_GeminiResponseWrapper = namedtuple("GeminiOcrResponse", ["text", "num_output_tokens"])


class LlmDocumentParser:
    """
    A parser for extracting elements from PDF documents, including text and tables.

    Attributes:
    """

    _verbose: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        max_requests_per_minute: int = 1000,
        model: Literal["gemini-1.5-flash"]
        | Literal["gemini-1.5-pro"]
        | str = "gemini-1.5-flash",
    ) -> None:
        gemini_sdk.configure(api_key=api_key)
        self.model = gemini_sdk.GenerativeModel(model_name=model)
        self._max_requests_per_minute = max_requests_per_minute

    async def _ocr_img(
        self, image: Image.Image, ocr_prompt: str
    ) -> _GeminiResponseWrapper:
        resp = await self.model.generate_content_async([ocr_prompt, image])

        finish_reason = resp.candidates[0].finish_reason

        # I hate google so much
        if finish_reason == 4:
            raise RecitationError()
        elif finish_reason == 1:  # STOP
            return _GeminiResponseWrapper(
                resp.text, resp.usage_metadata.total_token_count
            )
        else:
            raise ValueError(f"Unexpected finish reason: {finish_reason}")

    async def _submit_batch(
        self, images: List[Image.Image], ocr_prompt: str
    ) -> List[_GeminiResponseWrapper]:
        tasks = [self._ocr_img(image, ocr_prompt) for image in images]
        responses = await asyncio.gather(*tasks)
        return responses

    async def parse(
        self,
        file: Union[str, Path],
        ocr_prompt: str = PARSE_DOC_PROMPT,
    ) -> LlmParsedDocument:
        pdf_obj = Pdf(file)
        images = pdf_obj.to_imgs()
        bboxes = _get_content_bounding_boxes(pdf_obj)
        responses = []

        batch_size = self._max_requests_per_minute
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            start_time = time.monotonic()
            batch_responses = await self._submit_batch(batch, ocr_prompt)
            responses.extend(batch_responses)

            elapsed_time = time.monotonic() - start_time

            sleep_time = max(0, 60 - elapsed_time)
            if i + batch_size < len(images):
                await asyncio.sleep(sleep_time)

        pages: list[_LlmExtractedPage] = []
        for bbox, response in zip(bboxes, responses):
            page = _LlmExtractedPage(
                page_num=bbox.page,
                bbox=bbox,
                text=response.text,
                tokens=response.num_output_tokens,
            )
            pages.append(page)

        parsed_doc = LlmParsedDocument(
            pages=pages,
            filename=Path(file).name,
            num_pages=len(pages),
            coordinate_system=consts.COORDINATE_SYSTEM,
            table_parsing_kwargs={
                "model": self.model,
                "prompt": ocr_prompt,
            },
            creation_date=pdf_obj.file_metadata.get("creation_date"),
            last_modified_date=pdf_obj.file_metadata.get("last_modified_date"),
            last_accessed_date=pdf_obj.file_metadata.get("last_accessed_date"),
            file_size=pdf_obj.file_metadata.get("file_size"),
        )
        return parsed_doc


class BatchInProgress:
    def __init__(self, message: str = "Parsing is still in progress"):
        self.message = message

    def __str__(self) -> str:
        return self.message
