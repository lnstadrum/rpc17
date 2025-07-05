# Overview

Yet another "simple RPC library" for Python. Because the others are not simple enough.

Expose functions for remote execution using a decorator, then call them remotely as if they were members of a local object.

Serve your functions like this:
```python
import rpc17

@rpc17.expose
def make_it_emotional(message):
    """ A goofy function example """
    return f"{message}!!!"

# Serve all functions decorated with @rpc17.expose
rpc17.serve("tcp://localhost:8899")
```

Call your functions like that:
```python
from rpc17 import Remote

# Connect to the remote server
remote = Remote("tcp://localhost:8899")

# All the exposed functions are now accessible as members of `remote`:
result = remote.make_it_emotional("Hello")
print(result)   # 'Hello!!!'
```

# Installation

```bash
python3 -m pip install git+https://github.com/lnstadrum/rpc17.git
```

# Supported data types and function signatures

Functions accepting positional arguments and returning one or more of the following types are supported:

 - Core Python types (scalars, strings, booleans, `bytes`, `None`)
 - Lists and dictionaries containing core Python types (nesting is fine)
 - `numpy.ndarray`s

Tuples and sets are not really supported: they are converted to lists (and this is `msgspec`'s affair).

Keyword arguments are not supported.

Partial support for generators is included, with two caveats:
 - A generator must be fully consumed or explicitly closed by the client before invoking any other function on the same `Remote` instance (watch out for `strict=True` when using `zip(...)`).
 - Generation is asynchronous: the server begins yielding values before the client starts consuming them. Buffering occurs at the socket level—this can actually improve performance.

# Pros

 * Supports functions taking and returning `numpy.ndarray`s (in fact, designed for this particular purpose).
 * Exceptions are propagated and re-raised on the client side.
 * Efficient serialization with MessagePack
 * The server can handle multiple clients via threading or forking  (check out `rpc17.Server`).
 * Supports both TCP/IP and Unix domain sockets.

# Cons

 * Security/authentication? Never thought of it. Designed to be used in confined environments. Avoid exposing over the internet.
