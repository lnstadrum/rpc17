import enum
import functools
import msgspec
import numpy
import socket
import socketserver
import sys
from typing import Any, Iterable, Optional, Union

# Incremented every time a breaking change of the protocol happens.
PROTOCOL_VERSION = 1

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


# List of common exception types that can be raised on the client side
COMMON_EXCEPTIONS = {
    Exception: -1,
    NameError: -2,
    KeyError: -3,
    IndexError: -4,
    TypeError: -5,
    ValueError: -6
}


# List of functions that can be called remotely
FUNCTIONS_REGISTRY = []


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(tuple)


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
                        results = FUNCTIONS_REGISTRY[fun_num](*args)
                    except Exception as ex:
                        # Process the exception, if any
                        self._send_signed_int(COMMON_EXCEPTIONS.get(ex.__class__, -1))
                        self._send_bytes(_encode_value(str(ex)))
                        continue

                    # Send back the returned value
                    if isinstance(results, tuple):
                        self._send_signed_int(len(results))
                        for result in results:
                            self._send_bytes(_encode_value(result))
                    else:
                        self._send_signed_int(0)
                        self._send_bytes(_encode_value(results))
            except Server.ClientDisconnected:
                pass

    def __init__(self,
                 address: str = "localhost",
                 port: Optional[Union[Iterable[int], int]] = range(8899, 9100),
                 threading: Threading = Threading.THREADING):
        """ Creates a new server to serve functions remotely.

        Args:
            address (str, optional): Hostname or socket filename to bind the server to. Defaults to "localhost".
            port (Union[Iterable[int], int], optional): Port number or range of port numbers to bind the server to. Defaults to range(8899, 9100).
                                                        When set, TCP protocol is taken. Otherwise, when it is None, the Unix domain socket is used.
            threading (Threading, optional): Defines server behavior regarding multiple remote connections handling. Defaults to Threading.THREADING.
        """

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
        self.address = address
        if port is None:
            # using Unix domain socket
            self.server = server_class(address, Server.RequestHandler)
            self.port = None
        else:
            # using TCP protocol
            if isinstance(port, int):
                self.server = server_class((address, port), Server.RequestHandler)
                self.port = port
            else:
                for port_candidate in port:
                    try:
                        self.server = server_class((address, port_candidate), Server.RequestHandler)
                        self.port = port_candidate
                        break
                    except OSError:
                        pass
                else:
                    raise OSError(f"No available port in range {port}")

    def __enter__(self):
        self.server.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.__exit__(exc_type, exc_value, traceback)

    def serve_forever(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


class Remote:
    """ Connects to a server and allows to call functions remotely.
        Once connected, exposes the remote functions registry as its own member functions.
    """

    def _get_int(self, signed=False) -> int:
        return int.from_bytes(self.client.recv(4), "big", signed=signed)

    def _get_value(self):
        return _decode_value(self.client.recv(self._get_int(), socket.MSG_WAITALL))

    def __init__(self, address: str = "localhost", port: Optional[int] = 8899):
        """ Initiates connection to a remote server.

        Args:
            address (str, optional): Hostname or socket filename to connect to. Defaults to "localhost".
            port (int, optional): Port number to connect to. Defaults to 8899.
                                  When set, TCP protocol is taken. Otherwise, when it is None, the Unix domain socket is used.
        """
        if port is None:
            self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.client.connect(address)
        else:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((address, port))
        
        # send a magic header first
        self.client.sendall(MAGIC)
        check = self.client.recv(len(MAGIC), socket.MSG_WAITALL)
        if check != MAGIC:
            raise ConnectionError("Cannot connect to the server. "
                                  "The remote server is not a rpc17 server, "
                                  "or serves an incompatible protocol version.")

        # read out the list of available functions
        functions = _decoder.decode(self.client.recv(self._get_int()))
        for i, func in enumerate(functions):
            setattr(self, func, functools.partial(self.__call__, i))

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.client.__exit__(exc_type, exc_value, traceback)

    def __call__(self, func_number: int, *args):
        """ Calls a remote function by its number.
        """
        # prepare message header: number of the function to call and number of arguments
        header = ((func_number << 8) + len(args)).to_bytes(4, "big")
        self.client.sendall(header)

        # send arguments
        for arg in args:
            content = _encode_value(arg)
            self.client.sendall(len(content).to_bytes(4, "big") + content)

        # receive response
        return_code = self._get_int(signed=True)
        if return_code < 0:
            cls = next(cls for (cls, code) in COMMON_EXCEPTIONS.items() if return_code == code)
            raise cls(self._get_value())

        if return_code == 0:
            return self._get_value()

        return tuple(
            self._get_value()
            for _ in range(return_code)
        )


def serve(host: str = "localhost", port: Optional[int] = 8899):
    """ Serves the exposed functions to remote clients.
    """
    Server(host, port).serve_forever()
