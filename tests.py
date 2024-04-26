import unittest
import numpy as np

import transformations_and_projections as tp

class TestTransformInit(unittest.TestCase):
    def test_init_transform_as_unitary(self):
        obj = tp.Transform()
        self.assertTrue((obj.mat == np.identity(4)).all(),
                         "Transform matrix does not initialize to identity matrix.")

class TestTransformRotate(unittest.TestCase):
    def test_rotate_x_pi_quarter(self):
        tr = tp.Transform()
        u = np.array([1,0,0])
        theta = np.pi/4
        tr.rotate(theta,u)
        expected_result = np.array([
            [1,0            , 0             ,0],
            [0,np.cos(theta), -np.sin(theta),0],
            [0,np.sin(theta), np.cos(theta) ,0],
            [0,0            ,0              ,1]
        ])
        self.assertTrue(
            (tr.mat == expected_result).all(),
            "Rotation matrix for rot x by pi/4 is not as expected!",
        )

if __name__=="__main__":
    unittest.main()