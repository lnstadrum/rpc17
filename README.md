# Overview

Yet another "simple" RPC library for Python. Because the others are not simple enough.

Expose functions for remote execution with a decorator, then call them remotely as if they were members of a local object.

Serve your functions like this:
```python
import rpc17

@rpc17.expose
def make_it_emotional(message):
    """ A goofy function example """
    return f"{message}!!!"

# Serve all functions decorated with @rpc17.expose remotely
rpc17.serve(host="localhost", port=8890)
```

Call your functions like that:
```python
from rpc17 import Remote

# Connect to the remote server
remote = Remote("localhost", port=8890)

# All the exposed functions are now accessible as members of `remote`:
result = remote.make_it_emotional("Hello")
print(result)   # 'Hello!!!'
```

# Supported data types and function signatures

 - Python internals, mostly (scalars, strings, booleans, `bytes`, `None`)
 - Lists and dicts of Python internals (nesting is okay)
 - `numpy.ndarray`s

Tuples and lists are not really supported: they get converted to lists (and this is really `msgspec`'s affair).

Functions accepting positional arguments and returning one or more of the above are okay. Keyword arguments are not supported.

# Pros

 * Accepts functions taking and returning `numpy.ndarray`s (kinda designed for this particular purpose, actually).
 * Exceptions are rethrown remotely.
 * Efficient serialization with MessagePack

# Cons

 * Security/authentication? Never thought of it. Designed to be used in confined environments. Avoid exposing over internet.
