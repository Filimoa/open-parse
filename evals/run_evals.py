import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


SAMPLE_PDF_DIR = project_root / "evals/data"
EXPORT_DIR = project_root / "evals/parsed-data"

files = []
for file_path in SAMPLE_PDF_DIR.rglob("*.pdf"):
    files.append(file_path)

for file in files:
    pass
# run parsing on all them

# visualize the results

# write to a directory that's identical to the input directory
