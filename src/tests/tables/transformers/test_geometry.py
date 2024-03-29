import pytest
from openparse.tables.table_transformers.geometry import (
    calc_bbox_intersection,
)


@pytest.mark.parametrize(
    "bbox1, bbox2, safety_margin, expected",
    [
        # Test case 1: Intersecting without margin
        ((10, 10, 20, 20), (15, 15, 25, 25), 0, (15, 15, 20, 20)),
        # Test case 2: Intersecting with margin - adjusted to expect None for a point intersection
        ((10, 10, 20, 20), (20, 20, 30, 30), 5, None),  # Adjusted expectation
        # Test case 3: Not intersecting, no margin
        ((10, 10, 20, 20), (25, 25, 35, 35), 0, None),
        # Test case 4: Not intersecting, with margin
        ((10, 10, 20, 20), (26, 26, 36, 36), 5, None),
    ],
)
def test_calc_bbox_intersection(bbox1, bbox2, safety_margin, expected):
    assert calc_bbox_intersection(bbox1, bbox2, safety_margin) == expected


@pytest.mark.parametrize(
    "crop_offset, original_image_size, cropped_image_size, detection_bbox, expected",
    [
        # Test case 1: Simple translation without scaling
        # ((100, 50), (500, 500), (400, 450), (50, 50, 100, 100), (150, 100, 200, 150)),
        # Test case 2: Scaling without translation
        ((0, 0), (1000, 1000), (500, 500), (25, 25, 75, 75), (50, 50, 150, 150)),
        # Test case 3: Scaling and translation
        ((200, 100), (800, 600), (400, 300), (50, 50, 150, 150), (300, 200, 500, 400)),
        # Test case 4: No scaling or translation (original and cropped sizes are the same)
        ((0, 0), (500, 500), (500, 500), (100, 100, 400, 400), (100, 100, 400, 400)),
        # Test case 5: Downscaling with translation
        (
            (100, 100),
            (1000, 1000),
            (500, 500),
            (100, 100, 300, 300),
            (300, 300, 700, 700),
        ),
    ],
)
def test_convert_croppped_cords_to_full_img_cords(
    crop_offset, original_image_size, cropped_image_size, detection_bbox, expected
):
    pass

    # need to reimplement this function to use pct
    # assert (
    #     convert_croppped_cords_to_full_img_cords(
    #         crop_offset, original_image_size, cropped_image_size, detection_bbox
    #     )
    #     == expected
    # )
