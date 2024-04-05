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
    table_output_format: Literal["markdown", "html"]

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

