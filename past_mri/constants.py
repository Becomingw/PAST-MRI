from __future__ import annotations

FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
FIELD_VALUES = {
    "0.1T": 0.1,
    "1.5T": 1.5,
    "3T": 3.0,
    "5T": 5.0,
    "7T": 7.0,
}
FIELD_TO_ID = {name: idx for idx, name in enumerate(FIELDS)}
ID_TO_FIELD = {idx: name for name, idx in FIELD_TO_ID.items()}

CONTRASTS = ["T1W", "T2W", "T2FLAIR"]
CONTRAST_TO_ID = {name: idx for idx, name in enumerate(CONTRASTS)}
ID_TO_CONTRAST = {idx: name for name, idx in CONTRAST_TO_ID.items()}

TASK3_PAIRS = [(src, tgt) for src in FIELDS for tgt in FIELDS if src != tgt]

TASK3_VALIDATION_IDS = {
    "0.1T": ["0001", "0002", "0003"],
    "1.5T": ["0004", "0005", "0008"],
    "3T": ["0010", "0011", "0012"],
    "5T": ["0013", "0014", "0015"],
    "7T": ["0016", "0017", "0018"],
}

DEFAULT_Z_START = 150
DEFAULT_Z_END = 180
