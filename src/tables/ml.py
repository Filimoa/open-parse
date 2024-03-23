import logging
import time
from typing import Union, List, Optional, Literal, Self, Tuple, Any, Sequence

from PIL import Image  # type: ignore
import torch  # type: ignore
from torchvision import transforms  # type: ignore
from transformers import AutoModelForObjectDetection  # type: ignore
from transformers import TableTransformerForObjectDetection  # type: ignore

from .schemas import (
    _TableCellModelOutput,
    _TableModelOutput,
    Size,
    BBox,
    Table,
)
from src.utils import _crop_img_with_padding
from .geometry import _convert_table_cords_to_img_cords

t0 = time.time()

MIN_CONFIDENCE = 0.95
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

    tables = [t for t in detected_tables if t.confidence > MIN_CONFIDENCE]

    return tables


def get_table_content(page_dims: Size, img: Image.Image, table_bbox: BBox) -> Table:
    OFFSET = 10
    table_img = _crop_img_with_padding(img, table_bbox, padding=OFFSET)
    structure_id2label = {
        **structure_model.config.id2label,
        len(structure_model.config.id2label): "no object",
    }

    pixel_values_st = structure_transform(table_img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        outputs_st = structure_model(pixel_values_st)

    cells = _cell_outputs_to_objs(outputs_st, table_img.size, structure_id2label)

    for cell in cells:
        cell.bbox = _convert_table_cords_to_img_cords(
            crop_offset=(int(table_bbox[0] - OFFSET), int(table_bbox[1] - OFFSET)),
            original_image_size=img.size,
            cropped_image_size=img.size,
            detection_bbox=cell.bbox,
        )

    return Table.from_model_outputs(img.size, page_dims, table_bbox, cells)
