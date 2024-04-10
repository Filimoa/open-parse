### 1. Default Text Processing with PdfMiner
Use PdfMiner if your documents are text-heavy, well-structured, and do not contain non-text elements that require OCR.

```python
parser = openparse.DocumentParser()
parsed_doc = parser.parse(doc_path)
```

### 2. Optionally OCR with PyMuPDF
If your documents are scanned images or contain non-text elements, you may need to use OCR to extract text. PyMuPDF handles this, see their license [here](https://mupdf.com/licensing/index.html#commercial).

!!! warning "Use with caution"
    This method is not recommended as a default due to the additional computational cost and inherent inaccuracies of OCR.


```python hl_lines="2-4"
parser = openparse.DocumentParser()
parsed_doc = parser.parse(doc_path, ocr=True)
```

