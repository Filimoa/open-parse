Beyond accessing model attributes directly via their field names (e.g. parsed_content.text), models can be converted, dumped, serialized, and exported in a number of ways.

## Converting Models to Dictionaries
This is the primary way of converting a model to a dictionary.
```python 
parsed_content.model_dump()
```

## Converting Models to JSON
The `.model_dump_json()` method serializes a model directly to a JSON-encoded string that is equivalent to the result produced by .model_dump().

```python 
parsed_content.model_dump_json()
```
