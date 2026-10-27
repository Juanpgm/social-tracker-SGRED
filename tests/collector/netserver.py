"""A tiny scripted TCP server on loopback for fetcher tests that need REAL sockets and real time
(a trickling peer cannot be imitated with a mock transport). Each connection runs `script`."""

import contextlib
import socket
import threading
from collections.abc import Callable
from types import TracebackType

Script = Callable[[socket.socket, threading.Event], None]


def read_request(conn: socket.socket, stop: threading.Event) -> bytes:
    """Read until the end of the request head (or the peer or the test goes away)."""
    conn.settimeout(0.2)
    received = b""
    while b"\r\n\r\n" not in received and not stop.is_set():
        try:
            chunk = conn.recv(4096)
        except TimeoutError:
            continue
        except OSError:
            break
        if not chunk:
            break
        received += chunk
    return received


def trickle(conn: socket.socket, stop: threading.Event, data: bytes, every_s: float) -> None:
    """Send `data` one byte at a time, `every_s` apart, until it is gone or the test is over."""
    for index in range(len(data)):
        if stop.wait(every_s):
            return
        try:
            conn.sendall(data[index : index + 1])
        except OSError:
            return


class Server:
    def __init__(self, script: Script) -> None:
        self._script = script
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.1)
        self.port = self._listener.getsockname()[1]
        self.connections = 0
        self._conns: list[socket.socket] = []
        self._threads = [threading.Thread(target=self._accept_loop, daemon=True)]

    def url(self, path: str = "/feed", scheme: str = "http") -> str:
        return f"{scheme}://127.0.0.1:{self.port}{path}"

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            self._conns.append(conn)
            worker = threading.Thread(target=self._run, args=(conn,), daemon=True)
            self._threads.append(worker)
            worker.start()

    def _run(self, conn: socket.socket) -> None:
        try:
            self._script(conn, self._stop)
        except OSError:
            pass
        finally:
            conn.close()

    def __enter__(self) -> "Server":
        self._threads[0].start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._listener.close()
        for conn in self._conns:
            with contextlib.suppress(OSError):
                conn.close()
        for thread in self._threads:
            thread.join(timeout=2)
