## Llama Index 

We have a simple integration with Llama Index. You can convert the parsed document to Llama Index nodes and then create an index from those nodes.

```py
import openparse
from llama_index.core import VectorStoreIndex

doc_path = "./sample-docs/lyft-10k.pdf"
parser = openparse.DocumentParser()
parsed_doc = parser.parse(doc_path)

nodes = parsed_doc.to_llama_index_nodes()
index = VectorStoreIndex(nodes=nodes)
```

Now you can query the index 

```py
query_engine = index.as_query_engine()
response = query_engine.query("What do they do to make money?")
print(response)
```

You can also add nodes to an existing index

```py
existing_index.insert_nodes(nodes)
```
