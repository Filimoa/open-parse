import torch
from pydantic import BaseModel
from pathlib import Path
import sys
from openparse import consts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(consts.__file__).parent


class StructureModelConfig(BaseModel):
    weights_path: Path = root / "weights/unitable/unitable_large_structure.pt"
    vocab_path: Path = root / "weights/unitable/vocab_html.json"
    max_seq_len: int = 784


class BboxModelConfig(BaseModel):
    weights_path: Path = root / "weights/unitable/unitable_large_bbox.pt"
    vocab_path: Path = root / "weights/unitable/vocab_bbox.json"

    max_seq_len: int = 1024


class ContentModelConfig(BaseModel):
    weights_path: Path = root / "weights/unitable/unitable_large_content.pt"
    vocab_path: Path = root / "weights/unitable/vocab_cell_6k.json"
    max_seq_len: int = 200


class UniTableConfig(BaseModel):
    d_model: int = 768
    patch_size: int = 16
    nhead: int = 12
    dropout: float = 0.2

    structure: StructureModelConfig = StructureModelConfig()
    bbox: BboxModelConfig = BboxModelConfig()
    content: ContentModelConfig = ContentModelConfig()


def validate_weight_files_exist(config: UniTableConfig):
    weight_paths = [
        config.structure.weights_path,
        config.bbox.weights_path,
        config.content.weights_path,
    ]

    missing_files = [path for path in weight_paths if not path.exists()]

    if missing_files:
        print("Error: The following weight files are missing:", file=sys.stderr)
        for missing in missing_files:
            print(f"- {missing}", file=sys.stderr)
        sys.exit(1)


config = UniTableConfig()
validate_weight_files_exist(config)
