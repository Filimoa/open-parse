import pytest

from src.tables.table_transformers.geometry import (
    _calc_bbox_intersection,
    convert_croppped_cords_to_full_img_cords,
    # _convert_img_cords_to_pdf_cords,
)


# @pytest.mark.parametrize(
#     "bbox1, bbox2, safety_margin, expected",
#     [
#         # Test case 1: Intersecting without margin
#         ((10, 10, 20, 20), (15, 15, 25, 25), 0, (15, 15, 20, 20)),
#         # Test case 2: Intersecting with margin
#         ((10, 10, 20, 20), (20, 20, 30, 30), 5, (20, 20, 20, 20)),
#         # Test case 3: Not intersecting, no margin
#         ((10, 10, 20, 20), (25, 25, 35, 35), 0, None),
#         # Test case 4: Not intersecting, with margin
#         ((10, 10, 20, 20), (26, 26, 36, 36), 5, None),
#         # Test case 5: Intersecting with negative margin (margin should not make bboxes not intersect if they originally do)
#         ((10, 10, 20, 20), (15, 15, 25, 25), -5, (15, 15, 20, 20)),
#     ],
# )
# def test_calc_bbox_intersection(bbox1, bbox2, safety_margin, expected):
#     assert _calc_bbox_intersection(bbox1, bbox2, safety_margin) == expected


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


# @pytest.mark.parametrize(
#     "bbox, page_size, image_size, expected",
#     [
#         # Test case 1: Simple scaling up
#         ((10, 10, 20, 20), (2000, 2000), (100, 100), (200, 200, 400, 400)),
#         # Test case 2: Simple scaling down
#         ((200, 200, 400, 400), (1000, 1000), (2000, 2000), (100, 100, 200, 200)),
#         # Test case 3: Non-uniform scaling (different factors for x and y)
#         ((100, 50, 300, 150), (1000, 500), (400, 200), (250, 125, 750, 375)),
#         # Test case 4: No scaling (image size equals page size)
#         ((50, 50, 150, 150), (500, 500), (500, 500), (50, 50, 150, 150)),
#         # Test case 5: Scaling with non-square dimensions
#         ((10, 20, 30, 40), (800, 1600), (200, 400), (40, 80, 120, 160)),
#     ],
# )
# def test_convert_img_cords_to_pdf_cords(bbox, page_size, image_size, expected):
#     assert _convert_img_cords_to_pdf_cords(bbox, page_size, image_size) == expected
