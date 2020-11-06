import random
from types import GeneratorType

from src.app.colors.color_lookup import color_lookup

from typing import List, Tuple


class Colors:
    @staticmethod
    def as_list(length: int) -> List[List[int]]:
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
                if (
                    x := random.randint(0, len(color_lookup) - 1)
                ) not in existing:
                    short_fall.append(x)

            return existing + short_fall


# tail call optimisations: for fun!


class ColorsTailCallOpt:
    @staticmethod
    def as_list(length: int) -> List[List[int]]:
        indicies = (
            ColorsTailCallOpt._get_rand_indicies(length)
            if length < len(color_lookup)
            else ColorsTailCallOpt._get_rand_indicies(len(color_lookup))
        )
        return [list(color_lookup.values())[i] for i in indicies]

    @staticmethod
    def _get_rand_indicies(length: int, existing: List[int] = []) -> List[int]:
        def tco(tco_length: int, tco_existing: List[int] = []):
            if len(tco_existing) == tco_length:
                yield tco_existing
            else:
                short_fall = [
                    random.randint(0, len(color_lookup) - 1)
                    for i in range(tco_length - len(tco_existing))
                ]
                yield ColorsTailCallOpt._get_rand_indicies(
                    tco_length, list(set(tco_existing + short_fall))
                )

        def tramp(gen, *args, **kwargs):
            print("here", gen)
            g = gen(*args, **kwargs)
            return next(g) if isinstance(g, GeneratorType) else g

        return tramp(tco, length, existing)
