from __future__ import annotations

from .schemas import BBox, Size


def _calc_bbox_intersection(bbox1, bbox2, safety_margin=5.0):
    if safety_margin < 0:
        raise ValueError("Safety margin cannot be negative.")

    if (
        bbox1[2] <= bbox1[0]
        or bbox1[3] <= bbox1[1]
        or bbox2[2] <= bbox2[0]
        or bbox2[3] <= bbox2[1]
    ):
        raise ValueError("Bounding boxes must have non-zero width and height.")

    # Expand bounding boxes
    x1_expanded_min = min(bbox1[0], bbox2[0]) - safety_margin
    y1_expanded_min = min(bbox1[1], bbox2[1]) - safety_margin
    x2_expanded_max = max(bbox1[2], bbox2[2]) + safety_margin
    y2_expanded_max = max(bbox1[3], bbox2[3]) + safety_margin

    # Check if expanded boxes intersect
    if (
        x2_expanded_max <= max(bbox1[0], bbox2[0])
        or x1_expanded_min >= min(bbox1[2], bbox2[2])
        or y2_expanded_max <= max(bbox1[1], bbox2[1])
        or y1_expanded_min >= min(bbox1[3], bbox2[3])
    ):
        return None

    # Calculate and return the actual intersection based on original boxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Only return the intersection if it's valid
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)

    return None


def convert_img_cords_to_pdf_cords(
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


def convert_croppped_cords_to_full_img_cords(
    padding_pct: float,
    cropped_image_size: Size,
    detection_bbox: BBox,
    bbox: BBox,
) -> BBox:
    # Calculate the padding added around the cropped image
    cropped_width, cropped_height = cropped_image_size
    width_without_padding = cropped_width / (1 + 2 * padding_pct)
    height_without_padding = cropped_height / (1 + 2 * padding_pct)

    padding_x = (cropped_width - width_without_padding) / 2
    padding_y = (cropped_height - height_without_padding) / 2

    left, top, right, bottom = detection_bbox

    # Remove padding from the detection bbox
    left_adj = left - padding_x
    top_adj = top - padding_y
    right_adj = right - padding_x
    bottom_adj = bottom - padding_y

    # Add the original bbox's top-left corner to map back to original image coordinates
    orig_left, orig_top, _, _ = bbox
    left_adj += orig_left
    top_adj += orig_top
    right_adj += orig_left
    bottom_adj += orig_top

    return (left_adj, top_adj, right_adj, bottom_adj)
