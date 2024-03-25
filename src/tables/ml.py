import logging
import time
from typing import Union, List, Optional, Literal, Tuple, Any, Sequence

from PIL import Image  # type: ignore
import torch
from torchvision import transforms  # type: ignore
from transformers import AutoModelForObjectDetection  # type: ignore
from transformers import TableTransformerForObjectDetection  # type: ignore

from .schemas import (
    _TableCellModelOutput,
    _TableModelOutput,
    Size,
    BBox,
    _Table,
    _TableHeader,
    _TableRow,
    _TableHeaderCell,
    _TableDataCell,
)

# from .utils import crop_img_with_padding
from .geometry import (
    _convert_img_cords_to_pdf_cords,
    # _convert_table_cords_to_img_cords,
    _calc_bbox_intersection,
)

t0 = time.time()

MIN_TABLE_CONFIDENCE = 0.85
MIN_CELL_CONFIDENCE = 0.7
cuda_available = torch.cuda.is_available()
user_preferred_device = "cuda"
device = torch.device(
    "cuda" if cuda_available and user_preferred_device != "cpu" else "cpu"
)


class MaxResize:
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


detection_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection",
    revision="no_timm",
).to(device)

structure_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition",
    revision="no_timm",
).to(device)


detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


logging.info(f"Models loaded successfully 🚀: {time.time() - t0:.2f}s")


##################################
### === ML TABLE DETECTION === ###
##################################

# Adapted from:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Inference_with_Table_Transformer_(TATR)_for_parsing_tables.ipynb


def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Converts a bounding box format from center coordinates (cx, cy, width, height) to
    boundary coordinates (x_min, y_min, x_max, y_max).

    Parameters:
    - x: A tensor of shape (N, 4) representing N bounding boxes in cx, cy, w, h format.

    Returns:
    - A tensor of shape (N, 4) representing N bounding boxes in x_min, y_min, x_max, y_max format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def _rescale_bboxes(out_bbox: torch.Tensor, size: Size) -> torch.Tensor:
    """
    Rescales bounding boxes to the original image size.

    Parameters:
    - out_bbox: A tensor of bounding boxes in normalized format (relative to current size).
    - size: The target size (width, height) as a tuple of integers.

    Returns:
    - A tensor of rescaled bounding boxes in the target size.
    """
    width, height = size
    boxes = _box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes


def _outputs_to_objects(outputs: Any, img_size: Size, id2label: dict):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in _rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


def _cell_outputs_to_objs(
    outputs: Any, img_size: Size, id2label: dict
) -> List[_TableCellModelOutput]:
    clean_outputs = _outputs_to_objects(outputs, img_size, id2label)
    cells = []
    for cell in clean_outputs:
        cells.append(
            _TableCellModelOutput(
                label=cell["label"],
                confidence=cell["score"],
                bbox=cell["bbox"],
            )
        )
    return cells


def _table_outputs_to_objs(
    outputs: Any, img_size: Size, id2label: dict
) -> List[_TableModelOutput]:
    clean_outputs = _outputs_to_objects(outputs, img_size, id2label)
    tables = []
    for table in clean_outputs:
        tables.append(
            _TableModelOutput(
                label=table["label"],
                confidence=table["score"],
                bbox=table["bbox"],
            )
        )
    return tables


def find_table_bboxes(image: Image.Image) -> List[_TableModelOutput]:
    pixel_values = detection_transform(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        outputs = detection_model(pixel_values)

    detection_id2label = {
        **detection_model.config.id2label,
        len(detection_model.config.id2label): "no object",
    }

    detected_tables = _table_outputs_to_objs(outputs, image.size, detection_id2label)
    print(detected_tables)

    tables = [t for t in detected_tables if t.confidence > MIN_TABLE_CONFIDENCE]

    return tables


def table_from_model_outputs(
    image_size: Size,
    page_size: Size,
    table_bbox: BBox,
    table_cells: List[_TableCellModelOutput],
) -> "_Table":
    headers = [
        cell
        for cell in table_cells
        if cell.label == "table column header" and cell.confidence > MIN_CELL_CONFIDENCE
    ]
    rows = [
        cell
        for cell in table_cells
        if cell.label == "table row" and cell.confidence > MIN_CELL_CONFIDENCE
    ]
    cols = [
        cell
        for cell in table_cells
        if cell.label == "table column" and cell.confidence > MIN_CELL_CONFIDENCE
    ]

    header_objs = _preprocess_header_cells(headers, cols, image_size, page_size)
    row_objs = _process_row_cells(rows, cols, header_objs, image_size, page_size)

    return _Table(bbox=table_bbox, headers=header_objs, rows=row_objs)


def _preprocess_header_cells(
    header_rows: List[_TableCellModelOutput],
    cols: List[_TableCellModelOutput],
    image_size: Size,
    page_size: Size,
) -> List[_TableHeader]:
    header_cells = []
    for header in header_rows:
        header_row_cells = []
        for col in cols:
            cell_bbox = _calc_bbox_intersection(header.bbox, col.bbox, safety_margin=5)
            if cell_bbox:
                cell_bbox = _convert_img_cords_to_pdf_cords(
                    cell_bbox, page_size, image_size
                )
                header_row_cells.append(
                    _TableHeaderCell(
                        bbox=cell_bbox,
                    )
                )
        header_cells.append(_TableHeader(cells=header_row_cells))
    return header_cells


def _process_row_cells(
    rows: List[_TableCellModelOutput],
    cols: List[_TableCellModelOutput],
    headers: List[_TableHeader],
    image_size: Size,
    page_size: Size,
) -> List[_TableRow]:
    """
    Process row cells by checking against header cells for overlaps and converting coordinates.
    """
    data_cells = []
    for row in rows:
        row_cells = []
        for col in cols:
            cell_bbox = _calc_bbox_intersection(row.bbox, col.bbox, safety_margin=5)
            if cell_bbox:
                cell_bbox_pdf = _convert_img_cords_to_pdf_cords(
                    cell_bbox, page_size, image_size
                )
                if not _is_overlapping_with_headers(cell_bbox_pdf, headers):
                    row_cells.append(
                        _TableDataCell(
                            bbox=cell_bbox_pdf,
                        )
                    )
        if row_cells:
            data_cells.append(_TableRow(cells=row_cells))
    return data_cells


def _is_overlapping_with_headers(cell_bbox: BBox, headers: List[_TableHeader]) -> bool:
    """
    Some rows are also headers, we need to drop these. Check if a given cell's bounding box overlaps with any of the header cells' bounding boxes.
    """
    for header in headers:
        for hcell in header.cells:
            if (
                cell_bbox[0] < hcell.bbox[2]
                and cell_bbox[2] > hcell.bbox[0]
                and cell_bbox[1] < hcell.bbox[3]
                and cell_bbox[3] > hcell.bbox[1]
            ):
                return True
    return False


def crop_img_with_padding(
    image: Image.Image, bbox: BBox, padding_pct: float
) -> Image.Image:
    """
    Adds whitespace outside the image. Recomennded by the model authors.
    """
    if padding_pct < 0:
        raise ValueError("Padding percentage must be non-negative")
    if padding_pct >= 1:
        raise ValueError("Padding percentage must be less than 1")

    left, top, right, bottom = map(int, bbox)

    if not (0 <= left < right <= image.width) or not (
        0 <= top < bottom <= image.height
    ):
        raise ValueError("Bounding box is out of the image boundaries")

    try:
        cropped_image = image.crop((left, top, right, bottom))

        width = right - left
        height = bottom - top
        padding_x = int(width * padding_pct)
        padding_y = int(height * padding_pct)

        new_width = width + 2 * padding_x
        new_height = height + 2 * padding_y

        padded_image = Image.new("RGB", (new_width, new_height), color="white")
        padded_image.paste(cropped_image, (padding_x, padding_y))

        return padded_image

    except Exception as e:
        raise ValueError(f"Failed to crop the image: {e}")


def _convert_table_cords_to_img_cords(
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


def _display_img(image: Image.Image, cells) -> None:
    """
    Used for debugging to visualize the detected cells on the cropped table image.
    """
    from PIL import ImageDraw
    from PIL import Image
    from IPython.display import display  # type: ignore

    cropped_table_visualized = image.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)

    # Draw each cell's bounding box on the cropped table image
    for cell in cells:
        draw.rectangle(cell.bbox, outline="red")
    display(cropped_table_visualized)


def get_table_content(page_dims: Size, img: Image.Image, table_bbox: BBox) -> _Table:
    OFFSET = 0.05
    table_img = crop_img_with_padding(img, table_bbox, padding_pct=OFFSET)
    structure_id2label = {
        **structure_model.config.id2label,
        len(structure_model.config.id2label): "no object",
    }

    pixel_values_st = structure_transform(table_img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        outputs_st = structure_model(pixel_values_st)

    cells = _cell_outputs_to_objs(outputs_st, table_img.size, structure_id2label)
    _display_img(table_img, cells)

    for cell in cells:
        cell.bbox = _convert_table_cords_to_img_cords(
            padding_pct=OFFSET,
            cropped_image_size=table_img.size,
            detection_bbox=cell.bbox,
            bbox=table_bbox,
        )

    _display_img(img, cells)

    return table_from_model_outputs(img.size, page_dims, table_bbox, cells)


[
    _TableCellModelOutput(
        label="table column",
        confidence=0.999382734298706,
        bbox=(
            409.1182556152344,
            18.184553146362305,
            465.20648193359375,
            292.5752258300781,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9814857840538025,
        bbox=(
            31.537229537963867,
            255.77252197265625,
            677.0238647460938,
            280.1510009765625,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.6439926028251648,
        bbox=(
            594.0137939453125,
            17.816659927368164,
            676.3006591796875,
            45.59571838378906,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9869138598442078,
        bbox=(
            31.513933181762695,
            86.68744659423828,
            677.0390014648438,
            109.90785217285156,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9921457171440125,
        bbox=(
            561.6201782226562,
            17.90152359008789,
            592.106201171875,
            292.5290832519531,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9980902075767517,
        bbox=(
            465.2112121582031,
            18.084674835205078,
            502.8302001953125,
            292.522216796875,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9783489108085632,
        bbox=(
            31.688499450683594,
            131.60687255859375,
            676.813232421875,
            155.28964233398438,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9993212223052979,
        bbox=(
            502.56365966796875,
            18.011526107788086,
            561.2072143554688,
            292.37091064453125,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9799436926841736,
        bbox=(
            31.656147003173828,
            63.41431427001953,
            676.8348999023438,
            87.67230987548828,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.8909280896186829,
        bbox=(
            31.872610092163086,
            46.724239349365234,
            676.794189453125,
            63.79704666137695,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9846616387367249,
        bbox=(
            31.45240020751953,
            109.2430648803711,
            677.126220703125,
            131.0580291748047,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9997294545173645,
        bbox=(
            31.898826599121094,
            18.22787857055664,
            132.4683380126953,
            292.260986328125,
        ),
    ),
    _TableCellModelOutput(
        label="table projected row header",
        confidence=0.47230610251426697,
        bbox=(
            31.690425872802734,
            147.32350158691406,
            677.0588989257812,
            160.98069763183594,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.6372016668319702,
        bbox=(
            31.43528175354004,
            161.4745330810547,
            676.6739501953125,
            189.16810607910156,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9872730374336243,
        bbox=(31.6616268157959, 222.84130859375, 677.0166015625, 253.96148681640625),
    ),
    _TableCellModelOutput(
        label="table projected row header",
        confidence=0.7978255748748779,
        bbox=(
            31.66183853149414,
            238.96958923339844,
            676.131591796875,
            253.20811462402344,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.998776376247406,
        bbox=(
            280.3031005859375,
            18.28728485107422,
            314.39886474609375,
            292.4723815917969,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9989901185035706,
        bbox=(
            226.16860961914062,
            18.25072479248047,
            280.9783630371094,
            292.4764404296875,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9993372559547424,
        bbox=(
            314.7613220214844,
            18.312959671020508,
            371.51019287109375,
            292.551025390625,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.5930360555648804,
        bbox=(
            191.77781677246094,
            17.212753295898438,
            225.18179321289062,
            56.3880500793457,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.5483957529067993,
        bbox=(
            503.6430358886719,
            17.67702865600586,
            563.9542236328125,
            56.87588119506836,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.8504523038864136,
        bbox=(
            31.938331604003906,
            17.70589828491211,
            676.673583984375,
            46.42475891113281,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9996814727783203,
        bbox=(
            593.5136108398438,
            18.04166030883789,
            654.210205078125,
            292.2181701660156,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.994773805141449,
        bbox=(
            192.1643524169922,
            18.254772186279297,
            225.48068237304688,
            292.2580261230469,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9919609427452087,
        bbox=(
            369.79791259765625,
            18.596107482910156,
            409.4604187011719,
            291.8369445800781,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.6053591966629028,
        bbox=(
            225.5073699951172,
            17.440813064575195,
            280.3886413574219,
            56.818660736083984,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9910372495651245,
        bbox=(
            654.5828857421875,
            17.728055953979492,
            677.0433959960938,
            292.3228759765625,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9590184092521667,
        bbox=(31.7935791015625, 279.93798828125, 677.029052734375, 292.1258850097656),
    ),
    _TableCellModelOutput(
        label="table column header",
        confidence=0.9455636143684387,
        bbox=(
            31.801301956176758,
            17.829347610473633,
            676.829345703125,
            47.2924919128418,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.7744234204292297,
        bbox=(
            31.862939834594727,
            153.59616088867188,
            676.5872802734375,
            166.96688842773438,
        ),
    ),
    _TableCellModelOutput(
        label="table projected row header",
        confidence=0.75791996717453,
        bbox=(
            31.229103088378906,
            189.74362182617188,
            676.7484741210938,
            203.9435272216797,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9994822144508362,
        bbox=(
            132.5382080078125,
            18.012187957763672,
            191.77781677246094,
            292.43853759765625,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9697877764701843,
        bbox=(
            31.66042137145996,
            201.34347534179688,
            676.7758178710938,
            222.99620056152344,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.516968309879303,
        bbox=(
            463.93902587890625,
            17.65279197692871,
            501.96832275390625,
            56.022300720214844,
        ),
    ),
    _TableCellModelOutput(
        label="table projected row header",
        confidence=0.7232506275177002,
        bbox=(31.676925659179688, 45.94401168823242, 677.2421875, 64.3797836303711),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.741328239440918,
        bbox=(
            32.01799392700195,
            17.924636840820312,
            132.15640258789062,
            55.9454231262207,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.7745314836502075,
        bbox=(
            32.08577346801758,
            17.688077926635742,
            676.4373168945312,
            32.14672088623047,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.8223623037338257,
        bbox=(
            31.566959381103516,
            184.5835418701172,
            676.645263671875,
            201.9062957763672,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.7005451917648315,
        bbox=(
            131.77801513671875,
            17.44886016845703,
            192.1834259033203,
            56.07195281982422,
        ),
    ),
    _TableCellModelOutput(
        label="table",
        confidence=0.999920129776001,
        bbox=(
            31.704601287841797,
            18.139142990112305,
            676.7794189453125,
            291.7262878417969,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.5901471972465515,
        bbox=(
            278.8243408203125,
            17.849679946899414,
            315.2529602050781,
            56.11637878417969,
        ),
    ),
]
