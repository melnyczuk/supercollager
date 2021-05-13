import os
import pickle
from test.utils import describe, it
from unittest import TestCase

import numpy as np
from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type:ignore

from src.adapter import Adapter
from src.app import App


class End2EndTestCase(TestCase):
    @describe
    def test_e2e(self):
        show = int(os.getenv("SHOW", 0))
        update = int(os.getenv("UPDATE", 0))

        e2e_dir = os.path.abspath("test/e2e")
        pickle_dir = os.path.abspath(f"{e2e_dir}/pickles")
        data_dir = os.path.abspath(f"{e2e_dir}/data")

        def _run(output, pickle_file):
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
            output = list(tqdm(App.segment(image)))
            _run(output, "segment.pb")

        @it
        def collages():
            image = Adapter.load(f"{data_dir}/otter.jpeg")
            output = list(tqdm(App.collage(image)))
            _run(output, "collage.pb")

        @it
        def masks():
            image = Adapter.load(f"{data_dir}/otter.jpeg")
            output = list(tqdm(App.masks(image)))
            _run(output, "masks.pb")

        @it
        def alpha_mattes():
            video = Adapter.video(f"{data_dir}/otter.mp4")
            clip = App.alpha_matte(video, keyframe_interval=2, gain=50)
            output = np.array(list(clip), dtype=np.uint8)
            _run(output, "alpha_matte.pb")


def _show(output):
    for out in output:
        Image.fromarray(out).show()
    return


def _update(output, path):
    with open(path, "wb") as fileObject:
        pickle.dump(output, fileObject)


def _assert_test(output, path):
    with open(path, "rb") as fileObject:
        correct = pickle.load(fileObject)
        for i, out in enumerate(output):
            np.testing.assert_array_equal(out, correct[i])
