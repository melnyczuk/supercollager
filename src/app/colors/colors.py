import random
from typing import List, Tuple

from src.app.colors.color_lookup import color_lookup

ColorType = Tuple[int, int, int]


class Colors:
    @staticmethod
    def pick() -> ColorType:
        return list(color_lookup.values())[_randIndex()]

    @staticmethod
    def as_list(length: int) -> List[ColorType]:
        return [
            list(color_lookup.values())[i]
            for i in (
                Colors._get_rand_indicies(length)
                if length < len(color_lookup)
                else Colors._get_rand_indicies(len(color_lookup))
            )
        ]

    @staticmethod
    def _get_rand_indicies(length: int, existing: List[int] = []) -> List[int]:
        if len(existing) == length:
            return existing
        else:
            short_fall: List[int] = []

            while len(short_fall) < length - len(existing):
                if (x := _randIndex()) not in existing:
                    short_fall.append(x)

            return existing + short_fall


def _randIndex() -> int:
    return random.randint(0, len(color_lookup) - 1)
