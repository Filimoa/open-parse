from __future__ import annotations
from typing import List, Union

from .schemas import BBox, Size


def _calc_bbox_intersection(
    bbox1: BBox, bbox2: BBox, safety_margin: float = 5.0
) -> Union[BBox, None]:
    """
    Calculate the intersection of two bounding boxes with a safety margin, but return the intersection based on the original bounding boxes.

    :param bbox1: First bounding box (x1, y1, x2, y2)
    :param bbox2: Second bounding box (x1, y1, x2, y2)
    :param margin: Safety margin to expand bounding boxes for intersection calculation
    :return: The original intersection bounding box (x1, y1, x2, y2), or None if no intersection occurs
    """
    x1_expanded = max(bbox1[0] - safety_margin, bbox2[0] - safety_margin)
    y1_expanded = max(bbox1[1] - safety_margin, bbox2[1] - safety_margin)
    x2_expanded = min(bbox1[2] + safety_margin, bbox2[2] + safety_margin)
    y2_expanded = min(bbox1[3] + safety_margin, bbox2[3] + safety_margin)

    if x2_expanded > x1_expanded and y2_expanded > y1_expanded:
        x1 = max(bbox1[0], bbox2[0], x1_expanded)
        y1 = max(bbox1[1], bbox2[1], y1_expanded)
        x2 = min(bbox1[2], bbox2[2], x2_expanded)
        y2 = min(bbox1[3], bbox2[3], y2_expanded)

        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    return None


def _convert_table_cords_to_img_cords(
    crop_offset: Size,
    original_image_size: Size,
    cropped_image_size: Size,
    detection_bbox: BBox,
) -> BBox:
    """
    Calculate the original coordinates of a detection in the cropped and resized image.

    :param cropped_bbox: Bounding box of the cropped area in the original image.
    :param crop_offset: Offset (top-left corner) of the cropped area.
    :param original_image_size: Size of the original image before cropping and resizing.
    :param cropped_image_size: Size of the image after cropping and before resizing.
    :param detection_bbox: Detected bounding box in the cropped and resized image.

    :return: Transformed bounding box coordinates in the context of the original image.
    """
    scale_x = original_image_size[0] / cropped_image_size[0]
    scale_y = original_image_size[1] / cropped_image_size[1]

    original_bbox = (
        detection_bbox[0] * scale_x + crop_offset[0],
        detection_bbox[1] * scale_y + crop_offset[1],
        detection_bbox[2] * scale_x + crop_offset[0],
        detection_bbox[3] * scale_y + crop_offset[1],
    )

    return original_bbox


def _convert_img_cords_to_pdf_cords(
    bbox: BBox,
    page_size: Size,
    image_size: Size,
) -> BBox:
    scale_x = page_size[0] / image_size[0]
    scale_y = page_size[1] / image_size[1]
    return (
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    )
