
While we've chosen sensible defaults, you can add custom processing functions to the `DocumentParser` class to further process the extracted data.

```python
from openparse import processing, Node
from typing import List


class CustomCombineTables(processing.ProcessingStep):
    """
    Let's combine tables that are next to each other
    """

    def process(self, nodes: List[Node]) -> List[Node]:
        new_nodes = []
        print("Combining concurrent tables")
        for i in range(len(nodes) - 1):
            if "table" in nodes[i].variant and "table" in nodes[i + 1].variant:
                new_node = nodes[i] + nodes[i + 1]
                new_nodes.append(new_node)
            else:
                new_nodes.append(nodes[i])

        return new_nodes


# add a custom processing step to the pipeline
custom_pipeline = processing.BasicIngestionPipeline()
custom_pipeline.append_transform(CustomCombineTables())

parser = openparse.DocumentParser(
    table_args={"parsing_algorithm": "pymupdf"}, processing_pipeline=custom_pipeline
)
custom_10k = parser.parse(meta10k_path)
```

Or you can create your own custom processing pipeline from scratch by extending the `IngestionPipeline` class.

```python
from openparse import processing, Node
from typing import List


class BasicIngestionPipeline(processing.IngestionPipeline):
    """
    A basic pipeline for ingesting and processing Nodes.
    """

    def __init__(self):
        self.transformations = [
            processing.RemoveTextInsideTables(),
            processing.RemoveFullPageStubs(max_area_pct=0.35),
        ]
```
