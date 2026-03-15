from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
from PIL import Image


def decode_base64_image(obs_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(obs_str)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha1(frame.tobytes()).hexdigest()
