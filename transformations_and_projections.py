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
        
        a = np.array([
            
        ])

        pass

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