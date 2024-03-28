<p align="center">
    <img src="https://sergey-filimonov.nyc3.digitaloceanspaces.com/open-parse/open-parse-with-text-tp-logo.webp" width="350" />
</p>
<br/>

Open-Parse streamlines the process of preparing complex documents for analysis by LLMs. Our goal is to expose state-of-the-art deep learning models with a few lines of code while also providing flexible heuristic options for faster, basic parsing.

### Highlights

- **🔍 Visually-Driven:** Open-Parse visually analyzes documents for superior LLM input, going beyond naive text splitting.
- **✍️ Markdown Support:** Basic markdown support for parsing headings, bold and italics.
- **📊 High-Precision Table Support:** Extract tables into clean Markdown formats with accuracy that surpasses traditional tools.
- **🛠️ Extensible:** Easily implement your own post-processing steps.
- **💡Intuitive:** Great editor support. Completion everywhere. Less time debugging.
- **🎯 Easy:** Designed to be easy to use and learn. Less time reading docs.

<br/>
<p align="center">
    <img src="https://sergey-filimonov.nyc3.digitaloceanspaces.com/open-parse/marked-up-doc-2.webp" width="250" />
</p>

## Example

```python
import openparse

basic_doc_path = "./sample-docs/mobile-home-manual.pdf"
parser = openparse.DocumentParser()
parsed_basic_doc = parser.parse(basic_doc_path)

for node in parsed_basic_doc.nodes:
    print(node)
```

**📓 Try the sample notebook** <a href="https://colab.research.google.com/drive/1Z5B5gsnmhFKEFL-5yYIcoox7-jQao8Ep?usp=sharing" class="external-link" target="_blank">here</a>

## Requirements

Python 3.8+

**Dealing with PDF's:**

- <a href="https://github.com/pdfminer/pdfminer.six" class="external-link" target="_blank">pdfminer.six</a> Fully open source.

**Extracting Tables:**

- <a href="https://github.com/pymupdf/PyMuPDF" class="external-link" target="_blank">PyMuPDF</a> has some table detection functionality. Please see their <a href="https://mupdf.com/licensing/index.html#commercial" class="external-link" target="_blank">license</a>.
- <a href="https://huggingface.co/microsoft/table-transformer-detection" class="external-link" target="_blank">Table Transformer</a> is a deep learning approach.
- <a href="https://github.com/poloclub/unitable" class="external-link" target="_blank">unitable</a> is a more recent deep learning approach that seems promising _(coming soon)_

## Installation

#### 1. Core Library

```console
pip install openparse
```

**Enabling OCR Support**:

PyMuPDF will already contain all the logic to support OCR functions. But it additionally does need Tesseract’s language support data, so installation of Tesseract-OCR is still required.

The language support folder location must be communicated either via storing it in the environment variable "TESSDATA_PREFIX", or as a parameter in the applicable functions.

So for a working OCR functionality, make sure to complete this checklist:

1. Install Tesseract.

2. Locate Tesseract’s language support folder. Typically you will find it here:

   - Windows: `C:/Program Files/Tesseract-OCR/tessdata`

   - Unix systems: `/usr/share/tesseract-ocr/5/tessdata`

3. Set the environment variable TESSDATA_PREFIX

   - Windows: `setx TESSDATA_PREFIX "C:/Program Files/Tesseract-OCR/tessdata"`

   - Unix systems: `declare -x TESSDATA_PREFIX= /usr/share/tesseract-ocr/5/tessdata`

**Note:** _On Windows systems, this must happen outside Python – before starting your script. Just manipulating os.environ will not work!_

#### 2. ML Table Detection (Optional)

This repository provides an optional feature to parse content from tables using the state-of-the-art Table Transformer (DETR) model. The Table Transformer model, introduced in the paper "PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents" by Smock et al., achieves best-in-class results for table extraction.

```console
pip install "open-parse[ml]"
```

## Documentation

_Coming Soon_
