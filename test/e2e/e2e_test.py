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
        image = Adapter.load(f"{e2e_dir}/otter.jpeg")
        video = Adapter.video(f"{e2e_dir}/otter.mp4")

        @it
        def segments():
            pickle_path = os.path.abspath(f"{e2e_dir}/segment.pb")
            output = [o.np for o in tqdm(list(App.segment(image)))]

            if show:
                _show(output)
            elif update:
                _update(output, pickle_path)
            else:
                _assert_test(output, pickle_path)

        @it
        def collages():
            pickle_path = os.path.abspath(f"{e2e_dir}/collage.pb")
            output = list(tqdm([App.collage(image).np]))

            if show:
                _show(output)
            elif update:
                _update(output, pickle_path)
            else:
                _assert_test(output, pickle_path)

        @it
        def segments_videos():
            pickle_path = os.path.abspath(f"{e2e_dir}/video.pb")
            clip = App.alpha_matte(video, keyframe_interval=2, gain=50)
            output = np.array(list(clip.iter_frames()), dtype=np.uint8)

            if show:
                _show(output)
            elif update:
                _update(output, pickle_path)
            else:
                _assert_test(output, pickle_path)


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
