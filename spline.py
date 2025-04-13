import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import scipy.interpolate
import numpy as np
import json

# Spline shape parameters
params = {
    "center": 0.415,
    "height": 0.376,
    "neck_width": 0.077,
    "neck_heigth": 0.174,
    "head_width": 0.262,
    "head_height": 0.301,
    "height_offset": -0.156
}

# Projective transform parameters (h33 fixed to 1)
projective = {
    "scale_x": 1.0,
    "shear_x": 0.0,
    "translate_x": 0.0,
    "shear_y": 0.0,
    "scale_y": 1.0,
    "translate_y": 0.0,
    "perspective_x": 0.0,
    "perspective_y": 0.0
}

slider_ranges = {
    # Shape params
    "center": (0.0, 1.0),
    "height": (0.0, 1.0),
    "neck_width": (0.0, 0.5),
    "neck_heigth": (0.0, 1.0),
    "head_width": (0.0, 1.0),
    "head_height": (0.0, 1.0),
    "height_offset": (-0.5, 0.5),

    # Projective params
    "scale_x": (0.1, 2.0),
    "scale_y": (0.1, 2.0),
    "shear_x": (-1.0, 1.0),
    "shear_y": (-1.0, 1.0),
    "translate_x": (-1.0, 1.0),
    "translate_y": (-1.0, 1.0),
    "perspective_x": (-1.0, 1.0),
    "perspective_y": (-1.0, 1.0)
}

# Generate control points from spline parameters
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

# Apply projective transformation to (x, y) points
def apply_projective(points, proj):
    H = np.array([
        [proj["scale_x"], proj["shear_x"], proj["translate_x"]],
        [proj["shear_y"], proj["scale_y"], proj["translate_y"]],
        [proj["perspective_x"], proj["perspective_y"], 1.0]
    ])
    pts = np.column_stack((points, np.ones(len(points))))
    pts_proj = pts @ H.T
    pts_proj /= pts_proj[:, 2][:, None]
    return pts_proj[:, :2]

# Interpolate spline
def interpolate_spline(pts):
    t = np.arange(len(pts))
    spline_x = scipy.interpolate.make_interp_spline(t, pts[:, 0])
    spline_y = scipy.interpolate.make_interp_spline(t, pts[:, 1])
    t_interp = np.linspace(t.min(), t.max(), 300)
    xs_interp = spline_x(t_interp)
    ys_interp = spline_y(t_interp)
    return np.column_stack([xs_interp, ys_interp])

# ---- Setup Main Plot Window ----
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.set_xlim(-1, 2)
ax1.set_ylim(-1, 2)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
[line] = ax1.plot([], [], 'b-')
[scat] = ax1.plot([], [], 'ro')
fig1.canvas.manager.set_window_title("Spline")

# ---- Setup Sliders Window ----
slider_defs = list(params.items()) + list(projective.items())
fig2, axs = plt.subplots(len(slider_defs), 1, figsize=(10, len(slider_defs)*0.3 + 1))
fig2.canvas.manager.set_window_title("Sliders")
if not isinstance(axs, (list, np.ndarray)):
    axs = [axs]

sliders = {}
for ax, (name, val) in zip(axs, slider_defs):
    vmin, vmax = slider_ranges.get(name, (0.0, 1.0))
    sliders[name] = Slider(ax, name, vmin, vmax, valinit=val)

# ---- Save / Load Buttons ----
button_ax_save = plt.axes([0.15, 0.01, 0.3, 0.04])
button_ax_load = plt.axes([0.55, 0.01, 0.3, 0.04])
btn_save = Button(button_ax_save, 'Save Spline')
btn_load = Button(button_ax_load, 'Load Spline')

def save_params(event):
    data = {"params": params, "projective": projective}
    with open("spline.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Spline saved to spline.json")

def load_params(event):
    try:
        with open("spline.json", "r") as f:
            data = json.load(f)
        for key in params:
            if key in data.get("params", {}):
                val = data["params"][key]
                params[key] = val
                sliders[key].set_val(val)
        for key in projective:
            if key in data.get("projective", {}):
                val = data["projective"][key]
                projective[key] = val
                sliders[key].set_val(val)
        update(None)
        print("Spline loaded from spline.json")
    except Exception as e:
        print(f"Failed to load: {e}")

btn_save.on_clicked(save_params)
btn_load.on_clicked(load_params)

# ---- Update Function ----
def update(val):
    for name in params:
        params[name] = sliders[name].val
    for name in projective:
        projective[name] = sliders[name].val

    points = np.array(generate_points(params))
    spline = interpolate_spline(points)
    spline_proj = apply_projective(spline, projective)
    points_proj = apply_projective(points, projective)  # still show transformed control points

    line.set_data(spline_proj[:, 0], spline_proj[:, 1])
    scat.set_data(points_proj[:, 0], points_proj[:, 1])
    ax1.relim()
    ax1.autoscale_view()
    fig1.canvas.draw_idle()

# Connect sliders
for s in sliders.values():
    s.on_changed(update)

update(None)
plt.show()
