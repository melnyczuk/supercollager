from dataclasses import dataclass
from .save import Save
from .load import Load


@dataclass(frozen=True)
class IO:
    load = Load
    save = Save
