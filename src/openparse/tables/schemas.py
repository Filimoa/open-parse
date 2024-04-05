from typing import List, Literal, Optional, Sequence, Tuple, Union

import fitz
from pydantic import BaseModel, model_validator

Size = Tuple[int, int]
BBox = Tuple[float, float, float, float]


