If you're using the ml dependencies, the config class manages the computational device settings for your project,

### Config

This is a simple wrapper around pytorch. Setting the device will fail if pytorch is not installed.

```python
import openparse

openparse.config.set_device("cpu")
```

Note if you're on apple silicon, setting this to `mps` runs significantly slower than on `cpu`.
