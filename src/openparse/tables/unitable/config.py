import torch
from pydantic import BaseModel
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StructureModelConfig(BaseModel):
    weights_path: Path = Path(
        "/Users/sergey/Developer/experiments/unitable/experiments/unitable_weights/unitable_large_structure.pt"
    )
    vocab_path: Path = Path(
        "/Users/sergey/Developer/business/open-parse/src/openparse/tables/unitable/vocab/vocab_html.json"
    )
    max_seq_len: int = 784


class BboxModelConfig(BaseModel):
    weights_path: Path = Path(
        "/Users/sergey/Developer/experiments/unitable/experiments/unitable_weights/unitable_large_bbox.pt"
    )
    vocab_path: Path = Path(
        "/Users/sergey/Developer/business/open-parse/src/openparse/tables/unitable/vocab/vocab_bbox.json"
    )
    max_seq_len: int = 1024


class ContentModelConfig(BaseModel):
    weights_path: Path = Path(
        "/Users/sergey/Developer/experiments/unitable/experiments/unitable_weights/unitable_large_content.pt"
    )
    vocab_path: Path = Path(
        "/Users/sergey/Developer/business/open-parse/src/openparse/tables/unitable/vocab/vocab_cell_6k.json"
    )
    max_seq_len: int = 200


class UniTableConfig(BaseModel):
    d_model: int = 768
    patch_size: int = 16
    nhead: int = 12
    dropout: float = 0.2

    structure: StructureModelConfig = StructureModelConfig()
    bbox: BboxModelConfig = BboxModelConfig()
    content: ContentModelConfig = ContentModelConfig()


config = UniTableConfig()
