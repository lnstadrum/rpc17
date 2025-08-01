import numpy
import rpc17
import os
import pytest
import tempfile
import types
from threading import Thread


@rpc17.expose
def echo(x):
    return x

@rpc17.expose
def triple(x):
    return x, x, x

@rpc17.expose
def raise_key_error():
    {1: 2}[3]

@rpc17.expose
def raise_custom_error():
    class CustomError(Exception): pass
    raise CustomError()

@rpc17.expose
def generate_integers(limit: int):
    for i in range(limit):
        yield i

@rpc17.expose
def generate_integers_nested(limit):
    for i in limit:
        yield generate_integers(i)


@pytest.fixture(scope="module")
def socket_filename():
    with tempfile.TemporaryDirectory() as tempdir:
        yield os.path.join(tempdir, "socket")


@pytest.fixture(scope="module", params=[range(8890, 9100), (None,)])
def server(request, socket_filename):
    for port in request.param:
        # choose server address depending on the port number value
        address = f"unix://{socket_filename}" if port is None else f"tcp://localhost:{port}"

        # attempt to start the server
        # if it fails, try the next port number in the range
        # if all ports are tried and none of them work, raise an error
        try:
            with rpc17.Server(address, threading=rpc17.Threading.SINGLE_THREADED) as server:
                thread = Thread(target=server.serve_forever)
                thread.start()

                yield server

                server.server.shutdown()
                thread.join()
                return
        except OSError:
            continue
    raise OSError("No available port in the range")


@pytest.fixture(scope="function")
def remote(server):
    with rpc17.Remote(server.address) as remote:
        yield remote


class TestFunctionsRegistry:
    def test(self):
        assert echo("123") == "123"
        assert triple(1) == (1, 1, 1)
        with pytest.raises(KeyError):
            raise_key_error()
        with pytest.raises(Exception):
            raise_custom_error()


class TestCommunication:
    def test_remote_function_availability(self, remote):
        assert remote.echo is not None
        assert remote.triple is not None
        assert remote.raise_key_error is not None
        assert remote.raise_custom_error is not None

    @pytest.mark.parametrize("value", [
        123,
        124.0,
        "125",
        b"126",
        True,
        None,
        {},
        [],
        {1: 2},
        [1, 2],
        [{}, {1: {1: 1, 2: None}, "a": "bc"}],
    ])
    def test_internal_types(self, remote, value):
        assert remote.echo(value) == value
        assert remote.triple(value) == (value, value, value)

    @pytest.mark.parametrize("shape", [(100, 100), (1000, 1000), (1, 2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64, numpy.uint8])
    def test_ndarray(self, remote, shape, dtype):
        value = numpy.random.uniform(size=shape).astype(dtype)
        assert numpy.all(remote.echo(value) == value)

        for x in remote.triple(value):
            assert numpy.all(x == value)

    def test_bad_number_of_args(self, remote):
        with pytest.raises(TypeError):
            remote.echo(1, 2, 3)


class TestExceptions:
    def test_key_error(self, remote):
        with pytest.raises(KeyError):
            remote.raise_key_error()

    def test_custom_error(self, remote):
        with pytest.raises(rpc17.RemoteException):
            remote.raise_custom_error()


class TestGenerator:
    def test_generator(self, remote):
        gen = remote.generate_integers(10)
        assert isinstance(gen, types.GeneratorType)
        for x, y in enumerate(gen):
            assert x == y

        # ensure that the generator can be iterated multiple times
        gen = remote.generate_integers(20)
        values = list(gen)
        assert values == list(range(20))

    def test_nested_generators(self, remote):
        limits = [2, 5, 3]
        gen = remote.generate_integers_nested(limits)
        assert isinstance(gen, types.GeneratorType)
        for lim, g in zip(limits, gen):
            for x, y in zip(range(lim), g):
                assert x == y
            g.close()

    def test_early_close(self, remote):
        # begin consuming a generator
        gen = remote.generate_integers(100)
        for _ in range(10):
            next(gen)

        # close it before fully consumed
        gen.close()

        # ensure a new generator instance can be created
        for x, y in zip(range(5), remote.generate_integers(5)):
            assert x == y
