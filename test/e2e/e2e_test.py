from test.utils import describe, it
from unittest import TestCase

import os
import pickle
import numpy as np

from src.app import App


class End2EndTestCase(TestCase):
    @describe
    def test_e2e(self):
        e2e_dir = os.path.abspath("test/e2e")

        @it
        def segments():
            output = App.segment([f"{e2e_dir}/otter.jpeg"])
            with open(
                os.path.abspath(f"{e2e_dir}/segment.pb"), "rb"
            ) as fileObject:
                correct = pickle.load(fileObject)

            output[0].img.pil.show()
            output[1].img.pil.show()

            np.testing.assert_array_equal(output[0].img.np, correct[0].img.np)
            np.testing.assert_array_equal(output[1].img.np, correct[1].img.np)

        @it
        def collages():
            output = App.collage([f"{e2e_dir}/otter.jpeg"])
            with open(
                os.path.abspath(f"{e2e_dir}/collage.pb"), "rb"
            ) as fileObject:
                correct = pickle.load(fileObject)

            output.pil.show()

            np.testing.assert_array_equal(output.np, correct.np)
