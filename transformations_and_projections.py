import numpy as np
from typing import Tuple

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
        Calculate the rotation matrix corresponding to clockwise
        rotation around axis u by theta radians and update the
        mat attribute accordingly.

        Args:
            theta(float): Angle of rotation in radians.
            u(numpy.ndarray): Axis of rotation
        """
        assert u.ndim == 1
        assert u.size == 3
        assert np.linalg.norm(u)==1
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
        Update the Transform.mat to apply a translation by a vector t.

        Args:
            t(np.ndarray): Translation vector
        """
        assert t.size == 3
        self.mat[0:3,3] = t    

    def transform_pts(self, pts: np.ndarray) -> np.ndarray:
        """
        Transform the specified points according to transformation
            matrix.
        
        Args:
            pts(numpy.ndarray): Points (N) to apply the transformation on.
                Matrix size = Nx3
        """
        assert pts.shape[1] == 3

        result = np.zeros(pts.shape)

        for i in range(pts.shape[0]):
            p_h = np.ones(4)
            p_h[0:3] = pts[i]
            res_h = np.matmul(self.mat,p_h)
            result[i] = res_h[0:3]

        return result

def world2view(
        pts:np.ndarray,
        R:np.ndarray,
        c0:np.ndarray,
) -> np.ndarray:
    """
    Transforms input points from world coordinate system to camera c.s..
    Args:
        pts(numpy.ndarray): N by 3 matrix with 3d points per row
        R(numpy.ndarray): 3 by 3 matrix representing rotation of 
            new c.s.
        c0(numpy.ndarray): 3d origin point of new cs
    Returns:
        transformed_points(numpy.ndarray): N by 3 matrix with the input
            point coordinates in the camera coordinate system.
    """
    ph = np.ones(4)
    transformation_matrix = np.ones((4,4))
    transformation_matrix[0:3,0:3] = R
    transformation_matrix[0:3,3] = np.matmul(-R.T,c0)
    transformed_pts = np.zeros(pts.shape)
    for i in range(pts.shape[0]):
        ph[0:3] = pts[i]
        tph = np.matmul(transformation_matrix,ph)
        transformed_pts[i,:] = tph[0:3]
    return transformed_pts

def lookat(
        eye:np.ndarray,
        up:np.ndarray,
        target:np.ndarray,
) -> Tuple[np.ndarray,np.ndarray]:
    """
    Calculates and returns a Tuple containing a rotation matrix and
        a translation vector needed to perform a transformation from
        wcs to ccs.
    Args:
        eye(numpy.ndarray): 3d vector representing camera center
        up(numpy.ndarray): 3d vector representing camera up orientation
        target(numpy.ndarray): 3d vector representing camera lens target
    Returns:
       Tuple[
            R(numpy.ndarray): 3 by 3 Rotation matrix
            t(numpy.ndarray): 3d Translation vector
        ]: Rotation and translation to be applied to transform from 
            wcs to c(amera)cs
    """
    assert eye.shape == (3,) , "'eye' should be a 3d vector"
    assert up.shape == (3,) , "'up' should be a 3d vector"
    assert np.linalg.norm(up) == 1, "'up' should be a unit vector"
    assert target.shape == (3,) , "'target' should be a 3d vector" 
    
    zc = (target - eye) / np.linalg.norm(target-eye)
    t = up - np.dot(up,zc)*zc
    yc = t/np.linalg.norm(t)
    xc = np.cross(yc,zc)
    R = np.zeros((3,3))
    R[:,0] = xc
    R[:,1] = yc
    R[:,2] = zc
    t = eye

    return R,t

def perspective_project(
        pts: np.ndarray,
        focal: float,
        R: np.ndarray,
        t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the specified 3d points on the image plane, according
        to a pinhole perspective projection model.
    Args:
        pts     (numpy.ndarray) :   N by 3 matrix of input points(point per row)
        focal   (float)         :   Distance from camera surface to pinhole
        R       (numpy.ndarray) :   3 by 3 camera cs rotation matrix
        t       (numpy.ndarray) :   3d camera cs origin point

    Returns:
        projected_points    (numpy.ndarray) : N by 2, 2d coordinates of
            N inpup points on the camera surface
    """
    pass

def rasterize(
        pts_2d:     np.ndarray,
        plane_w:    int,
        plane_h:    int,
        res_w:      int,
        res_h:      int,  
) -> np.ndarray:
    """
    Rasterizes the input 2d coordinates from the camera plane to image
        pixel coordinates
    Args:
        pts_2d  (numpy.ndarray): input 2d points on camera plane
        plane_w (int): camera plane width
        plane_h (int): camera plane height
        res_w   (int): output image resolution width
        res_h   (int): output image resolution height
    """
    pass

def render_object(
        v_pos:np.ndarray,
        v_clr:np.ndarray,
        t_pos_idx:np.ndarray,
        plane_h:int,
        plane_w:int,
        res_h:int,
        res_w:int,
        focal:float,
        eye:np.ndarray,
        up:np.ndarray,
        target:np.ndarray,
) -> np.ndarray:
    """
    Render the specified object as viewed from the specified camera.
    """
    pass