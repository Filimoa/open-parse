from pathlib import Path

import openparse


project_root = Path(__file__).resolve().parent.parent
SAMPLE_PDF_DIR = project_root / "evals/data"
EXPORT_DIR = project_root / "evals/parsed-data"

parser = openparse.DocumentParser()

for file_path in SAMPLE_PDF_DIR.rglob("*.pdf"):
    pdf = openparse.Pdf(file_path)
    parsed = parser.parse(file_path)

    relative_path = file_path.relative_to(SAMPLE_PDF_DIR)
    export_path = EXPORT_DIR / relative_path

    export_path.parent.mkdir(parents=True, exist_ok=True)

    if not parsed.nodes:
        continue

    pdf.export_with_bboxes(parsed.nodes, export_path)
    print(f"Exported {file_path} to {export_path}")

print("Done! 🌟")
