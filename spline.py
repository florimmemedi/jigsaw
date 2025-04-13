import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.interpolate
import numpy as np

# Initial parameters
params = {
    "center": 0.415,
    "height": 0.376,
    "neck_width": 0.077,
    "neck_heigth": 0.174,
    "head_width": 0.262,
    "head_height": 0.301,
    "height_offset": -0.156
}

# Generate control points based on parameters
def generate_points(p):
    def offset(y): return y + p["height_offset"]
    p_first = (0, 0)
    p_neck_left = (p["center"] - p["neck_width"]/2.0, offset(p["neck_heigth"]))
    p_head_left = (p["center"] - p["head_width"]/2.0, offset(p["head_height"]))
    p_center = (p["center"], offset(p["height"]))
    p_head_right = (p["center"] + p["head_width"]/2.0, offset(p["head_height"]))
    p_neck_right = (p["center"] + p["neck_width"]/2.0, offset(p["neck_heigth"]))
    p_last = (1, 0)
    return [p_first, p_neck_left, p_head_left, p_center, p_head_right, p_neck_right, p_last]

# Initial spline generation
points = generate_points(params)
t = np.arange(len(points))
xs = [x for x, _ in points]
ys = [y for _, y in points]
spline_x = scipy.interpolate.make_interp_spline(t, xs)
spline_y = scipy.interpolate.make_interp_spline(t, ys)
t_interp = np.linspace(t.min(), t.max(), 300)
xs_interp = spline_x(t_interp)
ys_interp = spline_y(t_interp)

# Main plot window
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
[line] = ax1.plot(xs_interp, ys_interp, color='blue')
[scat] = ax1.plot(xs, ys, 'ro')

# Slider window
fig2, axs = plt.subplots(len(params), 1, figsize=(4, len(params)*0.6))
fig2.canvas.manager.set_window_title("Sliders")
fig1.canvas.manager.set_window_title("Spline")

sliders = {}
for ax, (name, val) in zip(axs, params.items()):
    sliders[name] = Slider(
        ax,
        label=name,
        valmin=0.0 if name != "height_offset" else -0.5,
        valmax=1.0 if name != "height_offset" else 0.5,
        valinit=val
    )


# Update function
def update(val):
    for name in params:
        params[name] = sliders[name].val
    points = generate_points(params)
    t = np.arange(len(points))
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    spline_x = scipy.interpolate.make_interp_spline(t, xs)
    spline_y = scipy.interpolate.make_interp_spline(t, ys)
    xs_interp = spline_x(t_interp)
    ys_interp = spline_y(t_interp)
    line.set_data(xs_interp, ys_interp)
    scat.set_data(xs, ys)
    fig1.canvas.draw_idle()

# Connect all sliders
for s in sliders.values():
    s.on_changed(update)

plt.show()
