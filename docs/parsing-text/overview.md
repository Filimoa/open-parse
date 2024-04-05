Text processing is how we extract textual elements from within a doc and convert it to Markdown. The output are Nodes that represent distinct parts of the layout - like a heading or paragraph. There's two options for text processing: `pymupdf` and `table-transformers`.

Choosing between `PdfMiner` and `PyMuPDF` depends on the specific needs of your project:

### 1. PdfMiner (default)
Use PdfMiner if your documents are text-heavy, well-structured, and do not contain non-text elements that require OCR.

```python
parser = openparse.DocumentParser()
parsed_doc = parser.parse(doc_path)
```

### 2. PyMuPDF
Opt for PyMuPDF if you are working with mixed content documents, especially if they include scanned images or require text extraction from visual elements.

**Note:** PyMuPDF is more computationally expensive than PdfMiner and may require additional setup for OCR support.  It also has a more restrictive [license](https://mupdf.com/licensing/index.html#commercial).


```python hl_lines="2-4"
parser = openparse.DocumentParser(
    text_args={
        "parsing_algorithm": "pymupdf"
    }
)
parsed_doc = parser.parse(doc_path)
```

