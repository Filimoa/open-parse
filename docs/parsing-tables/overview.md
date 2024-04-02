The ability to automatically identify and extract tables from PDF documents is a highly sought-after feature.

This is an active area of research and we aim to expose the best available tools to the community. This is a blend of newer deep learning approaches and traditional bounding box-based methods. **Until we have a clear winner, we aim to be parsing algorithm agnostic.**

Currently, we support two methods for extracting tables from PDFs:

## 1. PyMuPDF

PyMuPDF is a Python binding for the MuPDF library, which is a lightweight PDF, XPS and e-book viewer.

With version 1.23.0, PyMuPDF has added table recognition and extraction facilities to its rich set of features.

We find it tends to work well on dense tables, with a relatively simple structure. It's also very fast.

### `PyMuPDFArgsDict`

```python
# Arguments follow the following schema
class PyMuPDFArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["pymupdf"]
    table_output_format: Literal["str", "markdown", "html"]
```

The following arguments are supported:

- `parsing_algorithm` specifies the library used for parsing, in this case, `pymupdf`.
- `table_output_format` specifies the format of the extracted tables.

### Example

In the following example, we parse a 10-K document and extract the tables in markdown format.

```py
# defining the parser (table_args is a dict)
parser = openparse.DocumentParser(
    table_args={
        "parsing_algorithm": "pymupdf",
        "table_output_format": "markdown"
    }
)

# ingesting the document
parsed_10k = parser.parse(meta10k_path)


```

## 2. Table Transformers

Table Transformers is a deep learning approach to table detection and extraction. It is part of the Hugging Face Transformers library.

!!! warning "ML Dependencies Required"
    To use this method, you will need to install the ml dependencies by running `pip install "openparse[ml]"`.

We find it works well on tables with more complex structures and significant whitespace.

```python
# Arguments follow the following schema
class TableTransformersArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["table-transformers"]
    min_table_confidence: float
    min_cell_confidence: float
    table_output_format: Literal["str", "markdown", "html"]

```

The following arguments are supported:

- `parsing_algorithm` specifies the library used for parsing, in this case, `table-transformers`.
- `min_table_confidence` specifies the minimum confidence score for a table to be extracted.
- `min_cell_confidence` specifies the minimum confidence score for a cell to be extracted.
- `table_output_format` specifies the format of the extracted tables.

### Example


```python
parser = openparse.DocumentParser(
    table_args={
        "parsing_algorithm": "table-transformers",
        "min_table_confidence": 0.8,
    }
)
parsed_doc2 = parser.parse(doc_with_tables_path)
```


## 3. Unitable

Unitable is a more recent deep learning approach to table detection and extraction. It's fairly lightweight and has impressive performance.

!!! warning "ML Dependencies Required"
    To use this method, you will need to install the ml dependencies by running `pip install "openparse[ml]"`.
    
We will be integrating this method soon!
