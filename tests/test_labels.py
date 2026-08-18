"""Guards the frozen label map in emorec.data.labels.

These assertions are deliberately literal (not "does it round-trip some
arbitrary mapping") because the whole point of freezing the label map is
that it can never silently drift — see the module docstring in labels.py.
"""

from emorec.data.labels import EMOTION_NAMES, EMOTION_TO_ID, ID_TO_EMOTION, NUM_CLASSES

EXPECTED_EMOTION_TO_ID = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
}


def test_emotion_to_id_matches_frozen_literal() -> None:
    assert EMOTION_TO_ID == EXPECTED_EMOTION_TO_ID


def test_id_to_emotion_is_true_inverse() -> None:
    assert ID_TO_EMOTION == {v: k for k, v in EMOTION_TO_ID.items()}
    for emotion, idx in EMOTION_TO_ID.items():
        assert ID_TO_EMOTION[idx] == emotion


def test_num_classes_is_seven() -> None:
    assert NUM_CLASSES == 7
    assert len(EMOTION_TO_ID) == NUM_CLASSES


def test_emotion_names_ordered_by_id() -> None:
    assert len(EMOTION_NAMES) == NUM_CLASSES
    for idx, name in enumerate(EMOTION_NAMES):
        assert EMOTION_TO_ID[name] == idx
