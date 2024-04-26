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

class TestTransformTranslate(unittest.TestCase):
    def test_translate_x_by1(self):
        tr = tp.Transform()
        t = np.array([1,0,0])
        tr.translate(t)
        expected_result = np.array([
            [1,0,0,1],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ])
        self.assertTrue((tr.mat == expected_result).all())

class TestTransformTransformPts(unittest.TestCase):
    def test_basic(self):
        starting_points = np.array([
            [1,0,0], #point 1
            [0,1,0], #point 2
            [0,0,1], #point 3
        ])
        #create transform
        tr = tp.Transform()
        rot_axis = np.array([1,0,0])
        rot_theta = np.pi
        translation_vector = np.array([1,1,1])
        tr.rotate(rot_theta,rot_axis)
        tr.translate(translation_vector)
        transformed_points = tr.transform_pts(starting_points)
        expected_result = np.array([
            [2,1,1],
            [1,0,1],
            [1,1,0],
        ])
        self.assertTrue((transformed_points == expected_result).all())
if __name__=="__main__":
    unittest.main()