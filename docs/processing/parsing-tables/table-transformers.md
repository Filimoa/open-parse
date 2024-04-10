Table Transformers is a deep learning approach to table detection and extraction. It is part of the Hugging Face Transformers library.

!!! warning "ML Dependencies Required"
    To use this method, you will need to install the ml dependencies by running `pip install "openparse[ml]"`.

We find it works well on tables with more complex structures and significant whitespace.

## Parameters

| Name                 | Type                                 | Description                                                                                                  | Default |
|----------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------|---------|
| `parsing_algorithm`  | `Literal["table-transformers"]`      | The library used for parsing, in this case, table-transformers.                                              | None    |
| `min_table_confidence` | `float`                             | The minimum confidence score for a table to be extracted.                                                    | None    |
| `min_cell_confidence` | `float`                             | The minimum confidence score for a cell to be extracted.                                                     | None    |
| `table_output_format` | `Literal["markdown", "html"]`       | The format of the extracted tables. Supports both markdown and html.                                         | None    |


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

