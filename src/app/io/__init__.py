from dataclasses import dataclass
from .save import Save as _save
from .load import Load as _load


@dataclass(frozen=True)
class IO:
    load = _load
    save = _save
