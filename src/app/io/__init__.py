from dataclasses import dataclass

from src.app.io.load import Load
from src.app.io.save import Save


@dataclass(frozen=True)
class IO:
    load = Load
    save = Save
