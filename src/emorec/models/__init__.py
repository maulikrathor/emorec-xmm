"""Model definitions: encoder wrappers, baselines, and the gated cross-attention fusion model."""

from emorec.models.baselines import AudioOnly, Concat, TextOnly
from emorec.models.encoders import AudioEncoder, TextEncoder
from emorec.models.fusion import MultimodalERC, Output
from emorec.models.registry import build_model

__all__ = [
    "AudioEncoder",
    "AudioOnly",
    "Concat",
    "MultimodalERC",
    "Output",
    "TextEncoder",
    "TextOnly",
    "build_model",
]
