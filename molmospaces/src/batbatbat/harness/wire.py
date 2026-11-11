"""Message encoding between the client and the policy server: msgpack, with numpy arrays written as raw bytes plus
dtype and shape in OpenPI's format (openpi_client; RoboLab's evaluator uses the same). Not msgpack-numpy's format,
which encodes arrays differently."""
from __future__ import annotations

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in "VOc":
            raise ValueError(f"unsupported dtype {value.dtype}")
        if isinstance(value, np.ndarray):
            return {b"__ndarray__": True, b"data": value.tobytes(),
                    b"dtype": value.dtype.str, b"shape": value.shape}
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot encode {type(value).__name__}")


def _decode(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise ValueError("unsupported array dtype")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"]).copy()
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack(value):
    return msgpack.packb(value, default=_encode)


def unpack(value):
    if not isinstance(value, bytes):
        raise ValueError("expected a binary msgpack frame")
    return msgpack.unpackb(value, object_hook=_decode, strict_map_key=False)

