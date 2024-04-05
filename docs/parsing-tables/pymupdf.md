PyMuPDF is a Python binding for the MuPDF library, which is a lightweight PDF, XPS and e-book viewer.

With version 1.23.0, PyMuPDF has added table recognition and extraction facilities to its rich set of features.

We find it tends to work well on dense tables, with a relatively simple structure. It's also very fast.

```python
# Arguments follow the following schema
class PyMuPDFArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["pymupdf"]
    table_output_format: Literal["markdown", "html"]
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
