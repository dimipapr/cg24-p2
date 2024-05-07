from matplotlib import pyplot as plot
import numpy as np
import transformations_and_projections as tp
import os

c="N"
if not os.path.isfile("demo.py"):
    print("You are not running the script from the project directory!")
    print("Random stuff might happen. Continue anyway? N/y")
    input(c)
    if c!='Y' and c!="y":
        exit    

data = np.load("hw2.npy", allow_pickle=True).item(0)

v_pos = data["v_pos"]
v_clr = data["v_clr"]
t_pos_idx = data["t_pos_idx"]
eye = data["eye"]
up = data["up"]
target = data["target"]
focal = data["focal"]
plane_w = data["plane_w"]
plane_h = data["plane_h"]
res_w = data["res_w"]
res_h = data["res_h"]
theta_0 = data["theta_0"]
rot_axis_0 = data["rot_axis_0"]
t_0 = data["t_0"]
t_1 = data["t_1"]
