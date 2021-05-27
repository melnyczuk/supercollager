import os
import pickle
from test.utils import describe, it
from typing import Iterable
from unittest import TestCase

import numpy as np
from PIL import Image

from src.adapter import Adapter
from src.app import App
from src.app.transform import Transform


class End2EndTestCase(TestCase):
    @describe
    def test_e2e(self):
        show = int(os.getenv("SHOW", 0))
        update = int(os.getenv("UPDATE", 0))

        e2e_dir = os.path.abspath("test/e2e")
        pickle_dir = os.path.abspath(f"{e2e_dir}/pickles")
        data_dir = os.path.abspath(f"{e2e_dir}/data")

        def _run(output: Iterable[np.ndarray], pickle_file: str) -> None:
            pickle_path = os.path.abspath(f"{pickle_dir}/{pickle_file}")
            if show:
                _show(output)
            elif update:
                _update(output, pickle_path)
            else:
                _assert_test(output, pickle_path)

        @it
        def segments():
            image = Adapter.load(f"{data_dir}/otter.jpeg")
            output = App.segment(image)
            _run(output, "segment.pb")

        @it
        def collages():
            image = Adapter.load(f"{data_dir}/otter.jpeg")
            output = App.collage(image, background=10)
            _run(output, "collage.pb")

        @it
        def masks():
            image = Adapter.load(f"{data_dir}/otter.jpeg")
            output = App.masks(image)
            _run(output, "masks.pb")

        @it
        def super_resolutes():
            image = (
                Transform.resize(img, (32, 32))
                for img in Adapter.load(f"{data_dir}/otter.jpeg")
            )
            output = App.super_resolution(
                image,
                device="cpu",
                dsize=(128, 128),
            )
            _run(output, "super_resolution.pb")

        @it
        def abstracts():
            image = Adapter.load(f"{data_dir}/otter.jpeg")
            output = App.abstracts(
                image,
                limit=1,
                n_segments=1,
                dsize=(64, 80),
            )
            _run(output, "abstracts.pb")

        @it
        def alpha_mattes():
            video = Adapter.video(f"{data_dir}/otter.mp4")
            clip = App.alpha_matte(*video, keyframe_interval=2, gain=50)
            output = np.array(list(clip), dtype=np.uint8)
            _run(output, "alpha_matte.pb")


def _show(output: Iterable[np.ndarray]) -> None:
    for out in output:
        Image.fromarray(out).show()
    return


def _update(output: Iterable[np.ndarray], path: str) -> None:
    with open(path, "wb") as fileObject:
        pickle.dump(output, fileObject)
    return


def _assert_test(output: Iterable[np.ndarray], path: str) -> None:
    with open(path, "rb") as fileObject:
        correct = pickle.load(fileObject)
        for i, out in enumerate(output):
            np.testing.assert_array_equal(out, correct[i])
    return
