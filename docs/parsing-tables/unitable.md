Unitable is a deep learning approach to table detection and extraction.  **It achieves state-of-the-art (SOTA) performance on four of the largest TR datasets.** If table accuracy is your primary concern, this is the method to use.

Full credit goes to [ShengYun (Anthony) Peng](https://github.com/ShengYun-Peng) and his team for open sourcing their research in a reproducible manner. You can find the original repository with full training code [here](https://github.com/poloclub/unitable). We choose to directly use a small subset of their package along with their pre-trained weights.

## Installation

!!! warning "ML Dependencies Required"
    To use this method, you will need to install the ml dependencies by running `pip install "openparse[ml]"`.

Once you have pip installed openparse, you will need to download the weights of the model seperately by running the following command.

```console
$ openparse-download
```

Which will download the weights. They're about 1.5GB in size.


## Usage

```python
# Arguments follow the following schema
class UnitableArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["unitable"]
    min_table_confidence: float
    table_output_format: Literal["html"]

```

The following arguments are supported:

- `parsing_algorithm` specifies the library used for parsing, in this case, `unitable`.
- `min_table_confidence` specifies the minimum confidence score for a table to be extracted. Default to 0.75.
- `table_output_format` specifies the format of the extracted tables. Currently only suport html.


### Example


```python
parser = openparse.DocumentParser(
    table_args={
        "parsing_algorithm": "unitable",
        "min_table_confidence": 0.8,
    }
)
parsed_doc = parser.parse(doc_with_tables_path)
```

## Limitations

- This method is very computationally expensive. We recommend using it on a GPU.
- We currently use the table-transformers model to detect table locations. This model is not perfect and may miss some tables or crop them incorrectly. This negatively impacts the performance of unitable. We're actively looking at more robust models.

