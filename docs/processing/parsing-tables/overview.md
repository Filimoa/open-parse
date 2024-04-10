Automatically identifying and extracting tables from PDF documents is a highly desirable feature that many people are looking for. It's an exciting and active area of research, and our goal is to provide the community with access to the most effective tools available. 

**By default this is turned off.** Parsing tables adds significant computational overhead, so we've made it optional.

We're expose both cutting-edge deep learning techniques with traditional bounding box-based methods. Our approach is designed to be flexible, allowing users to select the parsing algorithm that works best for their specific requirements.

At the moment, we offer three options for extracting tables from PDFs: `unitable`, `pymupdf`, and `table-transformer`. Each of these methods has its own unique advantages and limitations, so you can choose the one that aligns with your needs. 

```python hl_lines="2"
parser = openparse.DocumentParser(
    table_args={...}
)

# ingesting the document
parsed_10k = parser.parse(meta10k_path)
```


## Become a Contributor?

!!! note "Become a Contributor"

    - If you have experience with quantizing models or optimizing them for inference, we would love to hear from you! Unitable achieves **state-of-the-art performance** on table extraction, but it is computationally expensive. We are looking to optimize the model for inference and reduce the size of the model.
