"""The frozen MELD emotion label map.

CONTRACT — read this before touching this file:

`EMOTION_TO_ID` below is FROZEN. It is written as a literal dict, not derived
from `sorted()`, not derived from any data file, not computed at import time.
Every other module in this repo that needs an emotion->id or id->emotion
mapping MUST import `EMOTION_TO_ID`, `ID_TO_EMOTION`, `NUM_CLASSES`, or
`EMOTION_NAMES` from here — never re-derive a mapping locally (e.g. via
`sorted(set(labels))`), even if it happens to produce the same order on
today's data.

Reordering, renaming, or regenerating this mapping silently invalidates
every trained checkpoint (the model's output layer is indexed by these ids)
and every reported number in every results table. If the mapping must ever
change, that is an architecture-level decision requiring an ADR, not a code
edit.
"""

EMOTION_TO_ID: dict[str, int] = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
}

ID_TO_EMOTION: dict[int, str] = {v: k for k, v in EMOTION_TO_ID.items()}

NUM_CLASSES: int = 7

EMOTION_NAMES: list[str] = [ID_TO_EMOTION[i] for i in range(NUM_CLASSES)]
