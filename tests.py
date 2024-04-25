import unittest
import numpy as np

import transformations_and_projections as tp

class TestTransform(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    def tearDown(self) -> None:
        return super().tearDown()
    def test_init_transform_as_unitary(self):
        obj = tp.Transform()
        self.assertTrue((obj.mat == np.identity(4)).all(),
                         "Transform matrix does not initialize to identity matrix.")

if __name__=="__main__":
    unittest.main()