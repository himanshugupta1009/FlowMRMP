"""PyBullet wrapper that automatically supplies a physics client ID."""

from __future__ import annotations

import functools
import inspect
import os

import pybullet


class BulletClient:
    """Own one PyBullet connection and bind calls to its client ID."""

    def __init__(self, connection_mode=None, hostName=None, options=""):
        self._shapes = {}
        self._pid = os.getpid()
        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY, options=options)
            if self._client >= 0:
                return
            connection_mode = pybullet.DIRECT
        if hostName is None:
            self._client = pybullet.connect(connection_mode, options=options)
        else:
            self._client = pybullet.connect(
                connection_mode, hostName=hostName, options=options
            )

    def __del__(self):
        if getattr(self, "_client", -1) >= 0 and self._pid == os.getpid():
            try:
                pybullet.disconnect(physicsClientId=self._client)
                self._client = -1
            except pybullet.error:
                pass

    def __getattr__(self, name):
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        if name == "disconnect":
            self._client = -1
        return attribute
