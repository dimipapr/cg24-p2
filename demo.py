from matplotlib import pyplot as plot
import numpy as np
import transformations_and_projections as tp
import os

figsize = (9.6,3.3)

c="N"
if not os.path.isfile("demo.py"):
    print("You are not running the script from the project directory!")
    print("Random stuff might happen. Continue anyway? N/y")
    input(c)
    if c!='Y' and c!="y":
        exit    

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
data = np.load("hw2.npy", allow_pickle=True).item(0)

v_pos = data["v_pos"].T
v_clr = data["v_clr"]
t_pos_idx = data["t_pos_idx"]
eye = data["eye"][:,0]
up = data["up"][:,0]
target = data["target"][:,0]
focal = data["focal"]
plane_w = data["plane_w"]
plane_h = data["plane_h"]
res_w = data["res_w"]
res_h = data["res_h"]
theta_0 = data["theta_0"]
rot_axis_0 = data["rot_axis_0"]
t_0 = data["t_0"]
t_1 = data["t_1"]

#original image
img0 = tp.render_object(
    v_pos,
    v_clr,
    t_pos_idx,
    plane_h,
    plane_w,
    res_h,
    res_w,
    focal,
    eye,
    up,
    target
)
fig0,ax0 = plot.subplots(1,1)
fig0.suptitle("Original Image")
ax0.imshow(img0)
fig0.savefig("output/original.png")

#rotation
transform = tp.Transform()
transform.rotate(theta_0,rot_axis_0)
v_pos_1 = transform.transform_pts(v_pos)
img1 = tp.render_object(
    v_pos_1,
    v_clr,
    t_pos_idx,
    plane_h,
    plane_w,
    res_h,
    res_w,
    focal,
    eye,
    up,
    target
)
fig1,ax1 = plot.subplots(1,1)
title = f"Rotated by {theta_0} rads around {rot_axis_0} axis"
fig1.suptitle(title)
ax1.imshow(img1)
fig1.savefig("output/original_r.png")

#translation 0
transform.translate(t_0)
v_pos_2 = transform.transform_pts(v_pos)
img2 = tp.render_object(
    v_pos_2,
    v_clr,
    t_pos_idx,
    plane_h,
    plane_w,
    res_h,
    res_w,
    focal,
    eye,
    up,
    target
)
fig2,ax2 = plot.subplots(1,1)
title = f"Translated by {t_0}"
fig2.suptitle(title)
ax2.imshow(img2)
fig2.savefig("output/original_r_t.png")

#translation 1
transform.translate(t_1)
v_pos_3 = transform.transform_pts(v_pos)
img3 = tp.render_object(
    v_pos_3,
    v_clr,
    t_pos_idx,
    plane_h,
    plane_w,
    res_h,
    res_w,
    focal,
    eye,
    up,
    target
)
fig3,ax3 = plot.subplots(1,1)
title = f"Translated by {t_1}"
fig3.suptitle(title)
ax3.imshow(img3)
fig3.savefig("output/original_r_t_t.png")

plot.show()