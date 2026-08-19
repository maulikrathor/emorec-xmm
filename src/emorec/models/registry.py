"""Model name -> class dispatch, the single place `cfg.model.name` is interpreted."""

from omegaconf import DictConfig
from torch import nn

from emorec.models.baselines import AudioOnly, Concat, TextOnly
from emorec.models.fusion import MultimodalERC

_REGISTRY: dict[str, type[nn.Module]] = {
    "text_only": TextOnly,
    "audio_only": AudioOnly,
    "concat": Concat,
    "xattn": MultimodalERC,
}


def build_model(cfg: DictConfig, tokenizer_len: int, new_token_ids: list[int]) -> nn.Module:
    """Construct the model named by `cfg.model.name` (one of `"text_only"`, `"audio_only"`, `"concat"`, `"xattn"`).

    `tokenizer_len`/`new_token_ids` are passed straight through to every
    model class (some ignore them, e.g. `AudioOnly`) so the registry has one
    uniform call signature regardless of which model is being built.
    Raises `ValueError` naming the valid choices if `cfg.model.name` isn't
    in the registry — checked BEFORE any encoder is constructed, so an
    unknown name never triggers a network call.
    """
    name = cfg.model.name
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model name {name!r}; choices: {sorted(_REGISTRY)}")
    model_cls = _REGISTRY[name]
    return model_cls(cfg, tokenizer_len, new_token_ids)
