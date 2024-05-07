import numpy as np

def vector_interp(p1,p2,V1,V2,coord,dim):
    """
    Calculates and returns vector value V at point p(x,y) using linear interpolation
    between values V1 at p1 and V2 at p2. Interpolation is performed along given
    axis according to dim value (1 for x, 2 for y). Coord value represents the
    interpolation point coordinate along the given interpolation axis.

    Args:
    p1 (numpy.ndarray) : (2,) point 1 (x1,y1)
    p2 (numpy.ndarray) : (2,) point 2 (x2,y2)
    V1 (numpy.ndarray) : (3,) vector value at point 1
    V2 (numpy.ndarray) : (3,) vector value at point 2
    coord (int) : interpolation point coordinate along interpolation axis
    dim (int) : [1,2] dim=1(2), interpolation along x(y)

    Returns:
    V (numpy.ndarray) : (3,) interpolated value
    """

    denominator = p2[dim-1]-p1[dim-1]
    if denominator == 0:
        return (V1+V2)/2
    interpolation_factor = (coord-p1[dim-1])/denominator
    interpolated_value = V1 + interpolation_factor * (V2-V1)
    return interpolated_value

def f_shading(img, vertices, vcolors):
    """
    Performs a triangle shading with the average color of its vertices and returns
    the input image with the triangle rendered on top.
    """

    ret_img = np.copy(img)
    triangle_color = np.average(vcolors,0)

    sides = np.array([
        [0,1],
        [1,2],
        [2,0],
    ])

    sides_x = vertices[:,0][sides]
    sides_y = vertices[:,1][sides]
    s_ind = np.argsort(sides_y,1)
    sides_y_sorted = np.take_along_axis(sides_y,s_ind,1)
    sides_x_sorted = np.take_along_axis(sides_x,s_ind,1)
    sides_ymin = sides_y_sorted[:,0]
    sides_ymax = sides_y_sorted[:,1]
    sides_xmin = sides_x_sorted[:,0]
    ymin = np.min(sides_ymin)
    ymax = np.max(sides_ymax)

    dx = vertices[:,0][sides[:,1]] - vertices[:,0][sides[:,0]]
    dy = vertices[:,1][sides[:,1]] - vertices[:,1][sides[:,0]]
    invm = dx/dy

    horizontal_sides = np.isinf(invm)
    point_sides = np.isnan(invm)
    excluded_sides = np.logical_or(horizontal_sides,point_sides)
    non_excluded_sides = np.logical_not(excluded_sides)
    
    active_sides = np.array([False,False,False])
    active_sides = np.logical_and ((sides_ymin==ymin),non_excluded_sides)
    border_points_x = sides_xmin.astype(np.float64)

    for y in range(ymin,ymax+1):
        active_border_points_x = border_points_x[active_sides]
        try:
            sorted_abpx = np.sort(active_border_points_x)
            sorted_abpx = np.round(sorted_abpx).astype(np.int32)
            for x in range(sorted_abpx[0],sorted_abpx[-1]):
                ret_img[x,y] = triangle_color
        except:
            pass#print("no active border points")
        new_active_sides = (sides_ymin == y+1)
        removed_active_sides = (sides_ymax == y)
        active_sides = np.logical_or(active_sides, new_active_sides)
        active_sides = np.logical_and(active_sides,np.logical_not(removed_active_sides))
        active_sides = np.logical_and(active_sides, non_excluded_sides)
        active_border_points_x_calc_index = np.logical_and(active_sides, np.logical_not(new_active_sides))
        border_points_x[active_border_points_x_calc_index] += invm[active_border_points_x_calc_index]
        
    #paint horizontal sides
    if horizontal_sides.any():
        for side in sides[horizontal_sides]:
            x = vertices[:,0][side]
            y = vertices[:,1][side]
            for y in range(min(y),max(y)+1):
                for x in range(min(x),max(x)+1):
                    ret_img[x,y]=triangle_color
    
    #paint point sides
    if point_sides.any():
        for side in sides[point_sides]:
            ret_img[vertices[side[0]][0],vertices[side[0]][1]]=triangle_color

    return ret_img

def g_shading(img, vertices, vcolors):
    
    """
    Performs a triangle shading using linear interpolation
    between the triangle vertices to calculate the color values and returns
    the input image with the triangle rendered on top.
    """

    ret_img = np.copy(img)

    sides = np.array([
        [0,1],
        [1,2],
        [2,0],
    ])

    sides_x = vertices[:,0][sides]
    sides_y = vertices[:,1][sides]
    s_ind = np.argsort(sides_y,1)
    sides_y_sorted = np.take_along_axis(sides_y,s_ind,1)
    sides_x_sorted = np.take_along_axis(sides_x,s_ind,1)
    sides_sorted = np.take_along_axis(sides,s_ind,1)
    sides_ymin = sides_y_sorted[:,0]
    sides_ymax = sides_y_sorted[:,1]
    sides_xmin = sides_x_sorted[:,0]
    ymin = np.min(sides_ymin)
    ymax = np.max(sides_ymax)

    dx = vertices[:,0][sides[:,1]] - vertices[:,0][sides[:,0]]
    dy = vertices[:,1][sides[:,1]] - vertices[:,1][sides[:,0]]
    invm = dx/dy

    horizontal_sides = np.isinf(invm)
    point_sides = np.isnan(invm)
    excluded_sides = np.logical_or(horizontal_sides,point_sides)
    non_excluded_sides = np.logical_not(excluded_sides)
    
    active_sides = np.array([False,False,False])
    active_sides = np.logical_and ((sides_ymin==ymin),non_excluded_sides)
    border_points_x = sides_xmin.astype(np.float64)
    border_point_colors = np.ones((3,3))
    for y in range(ymin,ymax+1):
        active_sides_index = np.asarray(np.where(active_sides==True))[0]
        active_border_points_x = border_points_x[active_sides_index]
        for i in range(active_sides_index.shape[0]):
            index = active_sides_index[i]
            border_point_colors[index] = vector_interp(
                vertices[sides[index,0]],
                vertices[sides[index,1]],
                vcolors[sides[index,0]],
                vcolors[sides[index,1]],
                y,
                2,
            )



        try:
            sorted_abpx_index = np.argsort(active_border_points_x)
            sorted_abpx = active_border_points_x[sorted_abpx_index]
            sorted_abpx = np.round(sorted_abpx).astype(np.int32)
            sorted_bp_colors = border_point_colors[active_sides][sorted_abpx_index]
            for x in range(sorted_abpx[0],sorted_abpx[-1]):
                ret_img[x,y] = vector_interp(
                    np.array([sorted_abpx[0],y]),
                    np.array([sorted_abpx[-1],y]),
                    sorted_bp_colors[0],
                    sorted_bp_colors[-1],
                    x,
                    1,
                )
        except:
            pass#print("no active border points")
        new_active_sides = (sides_ymin == y+1)
        removed_active_sides = (sides_ymax == y)
        active_sides = np.logical_or(active_sides, new_active_sides)
        active_sides = np.logical_and(active_sides,np.logical_not(removed_active_sides))
        active_sides = np.logical_and(active_sides, non_excluded_sides)
        active_border_points_x_calc_index = np.logical_and(active_sides, np.logical_not(new_active_sides))
        border_points_x[active_border_points_x_calc_index] += invm[active_border_points_x_calc_index]
        
    #paint horizontal sides
    # if horizontal_sides.any():
    #     for side in sides[horizontal_sides]:
    #         x = vertices[:,0][side]
    #         y = vertices[:,1][side]
    #         for y in range(min(y),max(y)+1):
    #             for x in range(min(x),max(x)+1):
    #                 ret_img[x,y]=triangle_color
    
    #paint point sides
    # if point_sides.any():
    #     for side in sides[point_sides]:
    #         ret_img[vertices[side[0]][0],vertices[side[0]][1]]=triangle_color

    return ret_img
    
def render_img(faces,vertices,vcolors,depth,shading):
    face_depth = np.average(depth[faces],1)
    s_ind = np.argsort(-face_depth)
    face_depth = face_depth[s_ind]
    faces = faces[s_ind]

    img = np.ones((512,512,3))

    if shading == "flat":
        for i in range(faces.shape[0]):
            img = f_shading(img,vertices[faces[i]], vcolors[faces[i]])
    else:
        for i in range(faces.shape[0]):
            img = g_shading(img, vertices[faces[i]], vcolors[faces[i]])

    return img