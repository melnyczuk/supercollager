from dataclasses import dataclass

from .load import Load
from .save import Save


@dataclass(frozen=True)
class IO:
    load = Load
    save = Save
