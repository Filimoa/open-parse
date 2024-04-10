PyMuPDF is a Python binding for the MuPDF library, which is a lightweight PDF, XPS and e-book viewer.

With version 1.23.0, PyMuPDF has added table recognition and extraction facilities to its rich set of features.

We find it tends to work well on dense tables, with a relatively simple structure. It's also very fast.

### Parameters:
| Name                  | Type      | Description                                                                 | Default |
|-----------------------|-----------|-----------------------------------------------------------------------------|---------|
| parsing_algorithm     | `Literal['unitable']` | The library used for parsing, in this case, unitable.                       | None    |
| min_table_confidence  | `float`   | The minimum confidence score for a table to be extracted. Default to 0.75.  | 0.75    |
| table_output_format   | `Literal['html']` | The format of the extracted tables. Currently only support html.            | 'html'  |


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
