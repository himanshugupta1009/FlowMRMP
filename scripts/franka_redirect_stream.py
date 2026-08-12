"""Redirect a Python stream and its underlying C file descriptor temporarily.

PyBullet prints some URDF messages directly from native code. Redirecting only
``sys.stdout`` or ``sys.stderr`` does not suppress those messages, so this
context manager temporarily redirects the corresponding operating-system file
descriptor as well.

Adapted from VAMP: https://github.com/KavrakiLab/vamp
"""

from __future__ import annotations

import ctypes
import os
import sys
from types import TracebackType
from typing import IO


class RedirectStream:
    """Redirect one text stream to a file for the lifetime of a context."""

    def __init__(self, stream: IO[str] = sys.stdout, file: str = os.devnull):
        self.stream = stream
        self.file = file
        self._file: IO[str] | None = None
        self._saved_descriptor: int | None = None

    @staticmethod
    def _flush_c_stream(stream: IO[str]) -> None:
        """Flush the C stdio stream that corresponds to stdout or stderr."""
        try:
            stream_name = stream.name[1:-1]
            libc = ctypes.CDLL(None)
            libc.fflush(ctypes.c_void_p.in_dll(libc, stream_name))
        except (AttributeError, IndexError, OSError, ValueError):
            # Some notebook and test streams do not expose a native descriptor.
            pass

    def __enter__(self) -> "RedirectStream":
        self.stream.flush()
        self._file = open(self.file, "w+")
        self._saved_descriptor = os.dup(self.stream.fileno())
        os.dup2(self._file.fileno(), self.stream.fileno())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._flush_c_stream(self.stream)
        if self._saved_descriptor is not None:
            os.dup2(self._saved_descriptor, self.stream.fileno())
            os.close(self._saved_descriptor)
        if self._file is not None:
            self._file.close()
