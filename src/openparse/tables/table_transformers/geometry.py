from __future__ import annotations


def calc_bbox_intersection(bbox1, bbox2, safety_margin=5.0):
    if safety_margin < 0:
        raise ValueError("Safety margin cannot be negative.")

    if (
        bbox1[2] <= bbox1[0]
        or bbox1[3] <= bbox1[1]
        or bbox2[2] <= bbox2[0]
        or bbox2[3] <= bbox2[1]
    ):
        raise ValueError("Bounding boxes must have non-zero width and height.")

    # Expand each box by the safety margin, then intersect the expanded boxes,
    # so two boxes that are within `safety_margin` of each other are treated as
    # overlapping. This matches the tolerance used by ``Element.overlaps`` and
    # makes ``safety_margin`` actually take effect: expanding a box on both
    # sides shifts each edge of the intersection outward by the margin.
    x1 = max(bbox1[0], bbox2[0]) - safety_margin
    y1 = max(bbox1[1], bbox2[1]) - safety_margin
    x2 = min(bbox1[2], bbox2[2]) + safety_margin
    y2 = min(bbox1[3], bbox2[3]) + safety_margin

    # Only return the intersection if it's valid
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)

    return None
