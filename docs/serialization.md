Beyond accessing model attributes directly via their field names (e.g. parsed_content.text), models can be converted, dumped, serialized, and exported in a number of ways.

## Converting Models to Dictionaries
This is the primary way of converting a model to a dictionary.

```python 
parsed_content.dict()
```

## Converting Models to JSON
You can run the following to convert the results to a python dict that can be serialized to JSON.
```python
parsed_content.json()
```

Optionally you can also run `.model_dump_json()` which serializes the results directly to a JSON-encoded string.

```python 
parsed_content.model_dump_json()
```
