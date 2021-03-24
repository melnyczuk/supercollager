from typing import Iterable, List

import numpy as np


class Keyframing:
    @staticmethod
    def interpolate_frames(
        frames: Iterable[np.ndarray],
        duration: int,
        k_interval: int,
    ) -> List[np.ndarray]:
        head, *middle, tail = frames
        output: List[np.ndarray] = []
        final_interval = duration % k_interval
        for frame in middle:
            output += Keyframing.__interpolate_between(head, frame, k_interval)
            head = frame
        output += Keyframing.__interpolate_between(head, tail, final_interval)
        return output

    @staticmethod
    def __interpolate_between(
        prev: np.ndarray,
        next: np.ndarray,
        k_interval: int,
    ) -> List[np.ndarray]:
        return [
            prev * ((k_interval - i) / k_interval) + next * (i / k_interval)
            for i in range(k_interval)
        ]
