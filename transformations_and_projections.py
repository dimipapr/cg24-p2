import numpy as np

class Transform:
    """
    TODO some real docstring
    A class for preforming affine transformations
    
    Attributes:
        mat(numpy.ndarray): A 4 by 4 numpy array representing an
            affine transformation matrix
    """
    
    def __init__(self):
        """
        Initialize a transform object
        """
        self.mat = np.identity(4)

    def rotate(self, theta: float, u: np.ndarray) -> None:
        """
        Rotate the transformation matrix.

        Args:
            theta(float): Angle of rotation in radians.
            u(numpy.ndarray): Axis of rotation
        """
        (ux,uy,uz) = u
        a = np.array([
            [ux*ux, ux*uy, ux*uz],
            [uy*ux, uy*uy, uy*uz],
            [uz*ux, uz*uy, uz*uz],
        ])

        b = np.array([
            [0, -uz, uy],
            [uz, 0, -ux],
            [uy, ux, 0],
        ])
        R = (1-np.cos(theta))*a \
            + np.cos(theta)*np.identity(3) \
            +np.sin(theta)*b

        self.mat[0:3,0:3] = R

    def translate(self, t: np.ndarray) -> None:
        """
        Translate the transformation matrix.

        Args:
            t(np.ndarray): Translation vector
        """
        pass

    def transform_pts(self, pts: np.ndarray) -> np.ndarray:
        """
        Transform the specified points according to transformation
            matrix.
        
        Args:
            pts(numpy.ndarray): Points (N) to apply the transformation on.
                Matrix size = Nx3
        """
        pass