import builtins
import enum
import functools
import msgspec
import numpy
import socket
import socketserver
import sys
import traceback
import types
from typing import Any, Tuple

# Incremented every time a breaking change of the protocol happens.
PROTOCOL_VERSION = 2

# Initial message when establishing connection.
# Loosely ensures that the client and the server are speaking the same language.
MAGIC = b"wtfrpc" + PROTOCOL_VERSION.to_bytes(2, "big")


class Type(enum.IntEnum):
    """ Argument/returned value type specification used for serialization
    """
    ANY = 0     # any internal Python data type

    FLOAT16 = 1
    FLOAT32 = 2
    FLOAT64 = 3
    FLOAT128 = 4
    FLOAT256 = 5
    COMPLEX64 = 6
    COMPLEX128 = 7
    UINT8 = 8
    UINT16 = 9
    UINT32 = 10
    UINT64 = 11
    UINT128 = 12
    UINT256 = 13
    INT8 = 14
    INT16 = 15
    INT32 = 16
    INT64 = 17
    INT128 = 18
    INT256 = 19


# Maps numpy.ndarray types to the serialized type
TYPE_MAP = {
    numpy.float16: Type.FLOAT16,
    numpy.float32: Type.FLOAT32,
    numpy.float64: Type.FLOAT64,
    numpy.float128: Type.FLOAT128,
    numpy.complex64: Type.COMPLEX64,
    numpy.complex128: Type.COMPLEX128,
    numpy.uint8: Type.UINT8,
    numpy.uint16: Type.UINT16,
    numpy.uint32: Type.UINT32,
    numpy.uint64: Type.UINT64,
    numpy.int8: Type.INT8,
    numpy.int16: Type.INT16,
    numpy.int32: Type.INT32,
    numpy.int64: Type.INT64
}

INV_TYPE_MAP = { val.value: dtype
                 for dtype, val in TYPE_MAP.items() }


class ReturnCode:
    OK = 0
    EXCEPTION = -1
    GENERATOR_START = -2
    GENERATOR_STOP = -3


# List of functions that can be called remotely
FUNCTIONS_REGISTRY = []


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(tuple)


def _parse_address(address: str) -> Tuple[str, int]:
    # start the server
    if address.startswith("tcp://"):
        host, port = address[6:].split(":")
        try:
            return host, int(port)
        except ValueError:
            raise ValueError(f"Invalid port number in address {address}")

    if address.startswith("unix://"):
        path = address[7:]
        return path, None

    raise ValueError(f"Unable to infer protocol for address {address}. Valid addresses start with tcp:// and unix://")


def _encode_value(value: Any) -> bytes:
    """ Serializes an argument or a returned value.
    """
    if isinstance(value, numpy.ndarray):
        try:
            type = TYPE_MAP[value.dtype.type]
        except KeyError:
            raise NotImplementedError(f"Unsupported dtype: {value.dtype}")

        shape = value.shape
        content = value.tobytes()
        return _encoder.encode((type, shape, content))

    else:
        return _encoder.encode((Type.ANY, value))


def _decode_value(message: bytes) -> Any:
    """ Deserializes an argument or a returned value.
    """
    type, *rest = _decoder.decode(message)
    if type == Type.ANY:
        return rest[0]
    else:
        shape, content = rest
        return numpy.frombuffer(content, dtype=INV_TYPE_MAP[type]).reshape(shape)


def expose(func):
    """ A decorator to add a given function to the list of functions that can be called remotely.
    """
    FUNCTIONS_REGISTRY.append(func)
    return func


class Threading(enum.Enum):
    """ Defines server behavior regarding multiple remote connections handling.
    """
    # Will handle a single remote at a time.
    # Additional remote connections will block while the server is busy.
    SINGLE_THREADED = "single_thread"

    # Will handle multiple remote connections simultaneously using threading.
    THREADING = "threading"

    # Will handle multiple remote connections simultaneously using forking.
    FORKING = "forking"


class Server:
    """ Serves the functions to remote clients
    """
    class ClientDisconnected(Exception): pass

    class RequestHandler(socketserver.BaseRequestHandler):
        def _get_int(self) -> int:
            reply = self.request.recv(4)
            if not reply:
                raise Server.ClientDisconnected()
            return int.from_bytes(reply, "big")

        def _send_signed_int(self, value: int):
            self.request.sendall(value.to_bytes(4, "big", signed=True))

        def _send_bytes(self, bytes: bytes):
            self.request.sendall(len(bytes).to_bytes(4, "big") + bytes)

        def _send_exception(self, ex: Exception):
            """ Sends the exception class, message and traceback.
            """
            class_name, message = ex.__class__.__name__.encode(), str(ex).encode()
            self._send_signed_int(ReturnCode.EXCEPTION)
            self._send_bytes(class_name)
            self._send_bytes(message)

            # Send back the traceback without the topmost frame
            tb = traceback.format_tb(ex.__traceback__)
            tb = tb.pop(0)
            self._send_bytes("".join(tb).encode())

        def _send_returned_value(self, values: Any):
            """ Sends back the returned value from a function call.
            """
            if isinstance(values, tuple):
                self._send_signed_int(len(values))
                for value in values:
                    self._send_bytes(_encode_value(value))

            elif isinstance(values, types.GeneratorType):
                # Send the generated values one by one
                try:
                    self._send_signed_int(ReturnCode.GENERATOR_START)
                    for val in values:
                        self._send_returned_value(val)
                    self._send_signed_int(ReturnCode.GENERATOR_STOP)
                except Exception as ex:
                    self._send_exception(ex)

            else:
                self._send_signed_int(ReturnCode.OK)
                self._send_bytes(_encode_value(values))

        def handle(self):
            # Start with the magic header
            check = self.request.recv(len(MAGIC), socket.MSG_WAITALL)
            if check != MAGIC:
                return
            self.request.sendall(MAGIC)

            # Send the list of exposed functions to the client
            self._send_bytes(_encoder.encode(tuple(f.__name__ for f in FUNCTIONS_REGISTRY)))

            # Process remote calls
            try:
                while True:
                    call = self._get_int()
                    fun_num = call >> 8
                    args_len = call & 0xFF

                    # Receive arguments
                    args = tuple(_decode_value(self.request.recv(self._get_int(), socket.MSG_WAITALL))
                                 for _ in range(args_len))

                    # Call the function
                    try:
                        if fun_num < 0 or fun_num >= len(FUNCTIONS_REGISTRY):
                            raise NameError(f"<function #{fun_num}>")
                        returned_value = FUNCTIONS_REGISTRY[fun_num](*args)

                        # Send back the returned value
                        self._send_returned_value(returned_value)
                    except Exception as ex:
                        self._send_exception(ex)
            except Server.ClientDisconnected:
                pass

    def __init__(self,
                 address: str = "tcp://localhost:8899",
                 threading: Threading = Threading.THREADING):
        """ Creates a new server to serve functions remotely.

        Args:
            address (str, optional): The address to serve at including protocol and port number if needed, e.g. tcp://localhost:8899 or unix://path/to/socket
            threading (Threading, optional): Defines server behavior regarding multiple remote connections handling. Defaults to Threading.THREADING.
        """

        self.address = address
        address, port = _parse_address(address)

        # Pick the server class
        server_class = ({
            Threading.SINGLE_THREADED: socketserver.TCPServer,
            Threading.THREADING: socketserver.ThreadingTCPServer,
            Threading.FORKING: socketserver.ForkingTCPServer
        }
        if port is not None else
        {
            Threading.SINGLE_THREADED: socketserver.UnixStreamServer,
            Threading.THREADING: socketserver.ThreadingUnixStreamServer,
            Threading.FORKING: socketserver.ForkingUnixStreamServer if sys.version_info >= (3, 12) else None
        }).get(threading)

        if server_class is None:
            raise ValueError(f"Unsupported threading mode {threading}")

        # Instantiate the server
        if port is None:
            # using Unix domain socket
            self.server = server_class(address, Server.RequestHandler)
        else:
            # using TCP protocol
            self.server = server_class((address, port), Server.RequestHandler)

    def __enter__(self):
        self.server.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.__exit__(exc_type, exc_value, traceback)

    def serve_forever(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


class RemoteException(Exception):
    pass


class Remote:
    """ Connects to a server and allows to call functions remotely.
        Once connected, exposes the remote functions registry as its own member functions.
    """

    def _get_int(self, signed=False) -> int:
        return int.from_bytes(self.client.recv(4), "big", signed=signed)

    def _get_value(self) -> Any:
        return _decode_value(self.client.recv(self._get_int(), socket.MSG_WAITALL))

    def _get_string(self) -> str:
        return self.client.recv(self._get_int(), socket.MSG_WAITALL).decode()

    def _return_generator(self):
        """ Returns a generator that yields values from the remote generator function.
        """
        try:
            while True:
                rc = self._get_int(signed=True)
                if rc == ReturnCode.GENERATOR_STOP:
                    return
                yield self._process_return_code(rc)
        except GeneratorExit:
            while rc != ReturnCode.GENERATOR_STOP:
                rc = self._get_int(signed=True)
                self._process_return_code(rc)

    def _process_return_code(self, rc: int):
        """ Processes the return code from the server.
            Reads the corresponding value from the socket and returns it.
        """
        if rc == ReturnCode.OK:
            # return a single value
            v = self._get_value()
            return v

        if rc > 0:
            # return a tuple of values
            return tuple(self._get_value()
                         for _ in range(rc))

        if rc == ReturnCode.EXCEPTION:
            # raise an exception
            class_name = self._get_string()
            message = self._get_string()
            traceback = self._get_string()
            exception_class = getattr(builtins, class_name, RemoteException)
            raise exception_class(traceback + message)

        if rc == ReturnCode.GENERATOR_START:
            # return a generator
            return self._return_generator()

        if rc == ReturnCode.GENERATOR_STOP:
            return

        raise NotImplementedError(f"Return code not implemented: {rc}.")

    def _connect(self):
        # establish a connection to the server
        if self.port is None:
            self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.client.connect(self.address)
        else:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.address, self.port))

        # send a magic header first
        self.client.sendall(MAGIC)
        check = self.client.recv(len(MAGIC), socket.MSG_WAITALL)
        if check != MAGIC:
            raise ConnectionError("Cannot connect to the server. "
                                  "The remote server is not a rpc17 server, "
                                  "or uses an incompatible protocol version.")

        # read out the list of available functions
        functions = _decoder.decode(self.client.recv(self._get_int()))
        for i, func in enumerate(functions):
            setattr(self, func, functools.partial(self.__call__, i))

    def __init__(self, address: str = "tcp://localhost:8899"):
        """ Initiates connection to a remote server.

        Args:
            address (str, optional): Hostname or socket filename to connect to. Defaults to "localhost".
        """
        self.address, self.port = _parse_address(address)
        self._connect()
        self._is_awaiting_response = False

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.client.__exit__(exc_type, exc_value, traceback)


    def __call__(self, func_number: int, *args, sync=True):
        """ Calls a remote function by its number.

        Args:
            func_number (int): Number of the remote function to call.
            *args: Arguments to pass to the remote function.
            sync (bool, optional): If True, waits for the return value before returning.
                                   Otherwise, returns immediately after the call.
                                   await_return() needs to called after that to get the result.
        """
        if self._is_awaiting_response:
            raise RuntimeError("Another call is in progress. Await for its return first.")

        # prepare message header: number of the function to call and number of arguments
        header = ((func_number << 8) + len(args)).to_bytes(4, "big")
        try:
            self.client.sendall(header)
        except BrokenPipeError:
            # try to reconnect once if the connection is broken
            self._connect()
            self.client.sendall(header)

        # send arguments
        for arg in args:
            content = _encode_value(arg)
            self.client.sendall(len(content).to_bytes(4, "big") + content)

        self._is_awaiting_response = True
        if sync:
            # wait for response
            return self.await_return()


    def await_return(self):
        """ Waits for a return value after a function call.
        """
        if not self._is_awaiting_response:
            raise RuntimeError("No function call in progress.")

        try:
            return_code = self._get_int(signed=True)

            # There might be pending exhausted generators:
            # ignore stop iteration return codes.
            while return_code == ReturnCode.GENERATOR_STOP:
                return_code = self._get_int(signed=True)

            return self._process_return_code(return_code)

        finally:
            self._is_awaiting_response = False


def serve(address):
    """ Serves the exposed functions to remote clients.
    """
    Server(address).serve_forever()


if __name__ == "__main__":
    import argparse
    import importlib
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Serves functions from given Python scripts or modules.")
    parser.add_argument("--threading", "-t",
                        default=Threading.THREADING.value,
                        choices=tuple(v.value for v in Threading),
                        help=f"defines server behavior regarding multiple remote connections handling. Defaults to '{Threading.THREADING.value}.'")
    parser.add_argument("file",
                        type=Path,
                        nargs="+",
                        help="a Python script or module to serve functions from")
    parser.add_argument("address",
                        help="The address to serve at including protocol and port number if needed, e.g. tcp://localhost:8899 or unix://path/to/socket")
    args = parser.parse_args()

    # loop the provided list of Python files
    for file_path in args.file:
        # execute them and extend the function registry
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        FUNCTIONS_REGISTRY += module.rpc17.FUNCTIONS_REGISTRY

    # start the server
    print(f"Serving at {args.address}...")
    Server(args.address, threading=Threading(args.threading)).serve_forever()
