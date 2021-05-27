import random
from typing import Iterable, List, Tuple

from src.app.colors.color_lookup import color_lookup

ColorType = Tuple[int, int, int]

colors: Tuple[ColorType] = tuple(color_lookup.values())


class Colors:
    @staticmethod
    def pick() -> ColorType:
        return colors[Colors.__randIndex()]

    @staticmethod
    def generate(length: int) -> Iterable[ColorType]:
        n = length if length < len(colors) else len(colors)
        return (colors[i] for i in Colors._get_rand_indicies(n))

    @staticmethod
    def _get_rand_indicies(
        length: int,
        existing: Iterable[int] = (),
    ) -> List[int]:
        arr = list(existing)
        while len(arr) < length:
            if (x := Colors.__randIndex()) not in arr:
                arr.append(x)
        return arr

    @staticmethod
    def __randIndex() -> int:
        return random.randint(0, len(color_lookup) - 1)
