import os
import pickle
from test.utils import describe, it
from unittest import TestCase

import numpy as np

from src.adapters.load import Load
from src.app import App


class End2EndTestCase(TestCase):
    @describe
    def test_e2e(self):
        show = int(os.getenv("SHOW", 0))
        update = int(os.getenv("UPDATE", 0))
        e2e_dir = os.path.abspath("test/e2e")
        inputs = Load.uris([e2e_dir])

        @it
        def segments():
            pickle_path = os.path.abspath(f"{e2e_dir}/segment.pb")
            output = App.segment(inputs)

            if show:
                _show(output)
            elif update:
                _update(output, pickle_path)
            else:
                _assert_test(output, pickle_path)

        @it
        def collages():
            pickle_path = os.path.abspath(f"{e2e_dir}/collage.pb")
            output = [App.collage(inputs)]

            if show:
                _show(output)
            elif update:
                _update(output, pickle_path)
            else:
                _assert_test(output, pickle_path)


def _show(output):
    for out in output:
        out.pil.show()
    return


def _update(output, path):
    with open(path, "wb") as fileObject:
        pickle.dump(output, fileObject)


def _assert_test(output, path):
    with open(path, "rb") as fileObject:
        correct = pickle.load(fileObject)
        for i, out in enumerate(output):
            np.testing.assert_array_equal(out.np, correct[i].np)
