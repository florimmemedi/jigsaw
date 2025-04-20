# Simulated Puzzle to solve

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
#plt.ion()  # turn on interactive mode
import cv2

import random
import math
import heapq
import time
import os
from functools import lru_cache

import numpy as np
import scipy

def print_grid(grid, orientation=None, states_explored=None, clear=True):
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')
    if states_explored is not None:
        print(f'Explored: {states_explored}')

    arrows = ['>', 'v', '<', '^']
    max_val = np.max(grid[grid >= 0]) if np.any(grid >= 0) else 0
    width = len(str(max_val)) + 1  # digits + 1 char for arrow or space

    for i, row in enumerate(grid):
        line = ""
        for j, cell in enumerate(row):
            if cell < 0:
                line += f"{'.':>{width}} "
            elif orientation is not None:
                arrow = arrows[orientation[i, j] % 4]
                line += f"{cell:{width - 1}}{arrow} "
            else:
                line += f"{cell:{width}} "
        print(line)


        
        
class VisualizerCV:
    def __init__(self, cell_size=64, upscale_factor=4):
        self.cell_size = cell_size
        self.upscale = upscale_factor
        self.drawn_cells = set()

    def showPuzzle(self, grid, orientations, pieces):
        n, m = grid.shape
        cs = self.cell_size * self.upscale
        h, w = n * cs, m * cs
        if not hasattr(self, 'canvas') or self.canvas.shape[:2] != (h, w):
            self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        else:
            self.canvas.fill(255)
        self.drawn_cells.clear()

        for i in range(n):
            for j in range(m):
                piece_id = grid[i, j]
                if piece_id < 0: continue
                if (i, j) in self.drawn_cells: continue
                piece_id = grid[i, j]
                usage_count = np.sum(grid == piece_id)
                self.draw_piece(i, j, pieces[piece_id], orientations[i, j], usage_count=usage_count)
                self.drawn_cells.add((i, j))

        # Downscale for display
        small = cv2.resize(self.canvas, (m * self.cell_size, n * self.cell_size), interpolation=cv2.INTER_AREA)
        cv2.imshow("Puzzle", small)
        cv2.waitKey(1)

    def draw_piece(self, row, col, piece, orientation, usage_count=1):
        cs = self.cell_size * self.upscale
        x = col * cs
        y = row * cs

        # Draw square
        bg_color = (200, 200, 200) if usage_count <= 1 else (180, 255, 180)
        cv2.rectangle(self.canvas, (x, y), (x + cs, y + cs), bg_color, thickness=-1)
        cv2.rectangle(self.canvas, (x, y), (x + cs, y + cs), (100, 100, 100), thickness=max(1, self.upscale))

        # Draw piece ID in center
        text = str(piece.id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = x + (cs - text_size[0]) // 2
        text_y = y + (cs + text_size[1]) // 2
        cv2.putText(self.canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.upscale, (0, 0, 0), max(1, self.upscale), lineType=cv2.LINE_AA)

        # Draw sides
        self.draw_side(x + cs//2, y, piece.top(orientation), 'top')
        self.draw_side(x + cs, y + cs//2, piece.right(orientation), 'right')
        self.draw_side(x + cs//2, y + cs, piece.bottom(orientation), 'bottom')
        self.draw_side(x, y + cs//2, piece.left(orientation), 'left')

    def draw_side(self, x_data, y_data, side, position):
        if side.spline is None:
            return

        color_map = {
            'top': (0, 0, 255),
            'bottom': (255, 0, 0),
            'left': (0, 255, 0),
            'right': (0, 165, 255)
        }
        color = color_map.get(position, (0, 0, 0))

        spline = side.spline.copy()
        center_x = 0.5
        spline[:, 0] = (spline[:, 0] - center_x) + center_x

        cs = self.cell_size * self.upscale
        spline[:, 0] *= cs
        spline[:, 1] *= cs

        if position in ['left', 'right']:
            spline = spline[:, [1, 0]]

        if not side.male:
            if position in ['right']:
                spline[:, 0] *= -1
            if position in ['bottom']:
                spline[:, 1] *= -1

        if side.male:
            if position in ['left']:
                spline[:, 0] *= -1
            if position in ['top']:
                spline[:, 1] *= -1

        if position == 'top':
            offset = np.array([x_data - cs / 2, y_data])
        elif position == 'bottom':
            offset = np.array([x_data - cs / 2, y_data])
        elif position == 'left':
            offset = np.array([x_data, y_data - cs / 2])
        elif position == 'right':
            offset = np.array([x_data, y_data - cs / 2])
        else:
            offset = np.array([x_data, y_data])

        spline += offset
        pts = spline.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(self.canvas, [pts], isClosed=False, color=color, thickness=max(2, self.upscale), lineType=cv2.LINE_AA)



class Visualizer:
    def __init__(self, cell_size=1):
        self.cell_size = cell_size
        self.drawn_cells = set()  # initialize once

    def showPuzzle(self, grid, orientations, pieces):
        n, m = grid.shape
        scale = 2  # inches per cell
        
        # Create a figure if none exists (allows incremental updates)
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(m * scale, n * scale))
            
        ax = self.ax
        ax.set_aspect('equal')
        ax.set_xlim(0, m * self.cell_size)
        ax.set_ylim(0, n * self.cell_size)
        ax.invert_yaxis()  # top-left origin
        ax.axis('off')
        
        for i in range(n):
            for j in range(m):
                piece_id = grid[i,j]
                
                if piece_id < 0: continue
                
                if (i, j) in self.drawn_cells:
                    continue  # already drawn
                    
                self.draw_piece(ax, i, j, pieces[piece_id], orientations[i, j])
                self.drawn_cells.add((i, j))

        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()

    def draw_piece(self, ax, row, col, piece, orientation):
        x = col * self.cell_size
        y = row * self.cell_size
        cs = self.cell_size

        # Draw the square for the piece
        rect = patches.Rectangle((x, y), cs, cs, linewidth=1, edgecolor='grey', facecolor='lightgrey')
        ax.add_patch(rect)

        # Draw side indicators (text or small lines)
        self.draw_side(ax, x + cs/2, y, piece.top(orientation), 'top')
        self.draw_side(ax, x + cs, y + cs/2, piece.right(orientation), 'right')
        self.draw_side(ax, x + cs/2, y + cs, piece.bottom(orientation), 'bottom')
        self.draw_side(ax, x, y + cs/2, piece.left(orientation), 'left')
        
        # Draw piece ID in the center
        show_orientation = False
        text = f'{str(piece.id)} ({str(orientation)})' if show_orientation else f'{str(piece.id)}'
        ax.text(x + cs / 2, y + cs / 2, text, ha='center', va='center', fontsize=8, color='black')

    def draw_side(self, ax, x_data, y_data, side, position):
        # Color by side
        side_colors = {
            'top': 'red',
            'bottom': 'blue',
            'left': 'green',
            'right': 'orange'
        }
        color = side_colors.get(position, 'black')
        
        
        # Draw spline if it exists
        if side.spline is not None:
            spline = side.spline.copy()

            # Parameters for visual gap (use 0.1 and 0.2 to best see both splines)
            edge_margin = 0.0 * self.cell_size
            shrink = 0.0  # percentage to shrink the spline horizontally (0.0–1.0)

            # Shrink x range inward (spline[:,0] in [0,1])
            center_x = 0.5
            spline[:, 0] = (spline[:, 0] - center_x) * (1 - shrink) + center_x

            # Scale to cell size
            spline[:, 0] *= self.cell_size
            spline[:, 1] *= self.cell_size

            # Positioning logic
            if position == 'top':
                offset = np.array([
                    x_data - self.cell_size / 2,
                    y_data + edge_margin
                ])
            elif position == 'bottom':
                offset = np.array([
                    x_data - self.cell_size / 2,
                    y_data - edge_margin
                ])
            elif position == 'left':
                spline = spline[:, [1, 0]]
                offset = np.array([
                    x_data + edge_margin,
                    y_data - self.cell_size / 2
                ])
            elif position == 'right':
                spline = spline[:, [1, 0]]
                offset = np.array([
                    x_data - edge_margin,
                    y_data - self.cell_size / 2
                ])
                
                
            if not side.male:
                if position in ["right"]:
                    spline[:, 0] *= -1
                    
                if position in ["bottom"]:
                    spline[:, 1] *= -1
                    
            if side.male:
                if position in ["left"]:
                    spline[:, 0] *= -1
                    
                if position in ["top"]:
                    spline[:, 1] *= -1
            
            spline += offset
            ax.plot(spline[:, 0], spline[:, 1], color=color, linewidth=1)


def generate_spline_from_params(p: dict, num_interp_points: int = 9):
    """Generate a spline based on shape parameters (as used in interactive tool)."""
    def offset(y): return y + p["height_offset"]

    points = [
        (0, 0),
        (p["center"] - p["neck_width"]/2.0, offset(p["neck_heigth"])),
        (p["center"] - p["head_width"]/2.0, offset(p["head_height"])),
        (p["center"], offset(p["height"])),
        (p["center"] + p["head_width"]/2.0, offset(p["head_height"])),
        (p["center"] + p["neck_width"]/2.0, offset(p["neck_heigth"])),
        (1, 0)
    ]

    points = np.array(points)
    t = np.arange(len(points))
    t_interp = np.linspace(t.min(), t.max(), num_interp_points)
    spline_x = scipy.interpolate.make_interp_spline(t, points[:, 0])
    spline_y = scipy.interpolate.make_interp_spline(t, points[:, 1])
    xs = spline_x(t_interp)
    ys = spline_y(t_interp)
    return np.stack((xs, ys), axis=1)


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


def apply_lens_distortion(points, k1=0.1, k2=0.01, k3=0.0, p1=0.0, p2=0.0):
    distorted = np.zeros_like(points)
    x = points[:, 0]
    y = points[:, 1]

    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_radial = x * radial
    y_radial = y * radial

    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    distorted[:, 0] = x_radial + x_tangential
    distorted[:, 1] = y_radial + y_tangential

    return distorted

def scale_to_fill_view(points, scale=1.0):
    # center around 0.5,0.5
    centered = points - 0.5
    return centered * scale + 0.5


def show_distortion_grid(grid_size=(3, 3), num_points=100):
    fig, axes = plt.subplots(*grid_size, figsize=(9, 9))
    plt.subplots_adjust(left=0.25, bottom=0.35)
    axes = axes.flatten()

    # Generate splines
    splines = [[Side.generate(num_points).spline for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    flat_splines = [spline for row in splines for spline in row]

    # Initial plots
    lines_orig = []
    lines_dist = []
    for ax, spline in zip(axes, flat_splines):
        l1, = ax.plot(spline[:, 0], spline[:, 1], 'b-', lw=1)
        l2, = ax.plot([], [], 'r-', lw=1)
        ax.set_aspect('equal')
        ax.axis('off')
        lines_orig.append(l1)
        lines_dist.append(l2)

    # Slider setup
    axcolor = 'lightgoldenrodyellow'
    slider_ax = {
        name: plt.axes([0.25, 0.25 - i * 0.04, 0.65, 0.03], facecolor=axcolor)
        for i, name in enumerate(["k1", "k2", "p1", "p2", "scale"])
    }

    sliders = {
        "k1": Slider(slider_ax["k1"], "k1", -1.0, 1.0, valinit=0.0),
        "k2": Slider(slider_ax["k2"], "k2", -1.0, 1.0, valinit=0.0),
        "p1": Slider(slider_ax["p1"], "p1", -0.2, 0.2, valinit=0.0),
        "p2": Slider(slider_ax["p2"], "p2", -0.2, 0.2, valinit=0.0),
        "scale": Slider(slider_ax["scale"], "scale", 0.01, 2.0, valinit=1.0),
    }

    def update(val):
        params = {k: sliders[k].val for k in ["k1", "k2", "p1", "p2"]}
        scale = sliders["scale"].val

        all_x, all_y = [], []

        for spline, l_dist in zip(flat_splines, lines_dist):
            spline = scale_to_fill_view(spline, scale)
            spline = apply_lens_distortion(spline, **params)
            spline = scale_to_fill_view(spline, 1.0/scale)
            l_dist.set_data(spline[:, 0], spline[:, 1])
            all_x.append(spline[:, 0])
            all_y.append(spline[:, 1])

        # Compute global bounding box
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        margin = 0.05
        xlim = (all_x.min() - margin, all_x.max() + margin)
        ylim = (all_y.min() - margin, all_y.max() + margin)

        for ax in axes:
            ax.set_xlim(0,1)
            ax.set_ylim(-0.1,1)

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    update(None)
    plt.show()

       
class Side:
    def __init__(self, spline: np.ndarray, male: bool, kind: str = "normal"):
        self.spline = spline
        self.male = male
        self.kind = kind  # "normal", "flat", or "unspecified"

    # apply slight deformation
    def copy(self):
        if self.spline is None:
            return Side(None, self.male, self.kind)
        
        spline = np.copy(self.spline)
        
        # small perturbations
        delta_e = 0.02
        
        projective = {
            "translate_x": 0.0,
            "translate_y": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "shear_x": np.random.uniform(-delta_e, delta_e),
            "shear_y": np.random.uniform(-delta_e, delta_e),
            "perspective_x": np.random.uniform(-delta_e, delta_e),
            "perspective_y": np.random.uniform(-delta_e, delta_e)
        }
        
        #spline = scale_to_fill_view(spline, scale=1.0)  # closer = larger scale
        spline = apply_lens_distortion(spline, k1=0.002, k2=0.005, p1=0.001, p2=0.001)
        spline = apply_projective(spline, projective)
        
        return Side(spline, self.male, self.kind)

    
    @staticmethod
    def generate(num_points: int = 100):
        """Generate a side with random center and height_offset; all other params fixed."""
        p = {
            "center": np.random.uniform(0.2, 0.8),
            "height": np.random.uniform(0.3, 0.5),#0.376,
            "neck_width": 0.077,
            "neck_heigth": np.random.uniform(0.1, 0.2),#0.174,
            "head_width": 0.262,
            "head_height": 0.301,
            "height_offset": np.random.uniform(-0.2, -0.1)
        }
        spline = generate_spline_from_params(p, num_interp_points=num_points)
        return Side(spline, random.choice([True, False]), kind="normal")


    @staticmethod
    def flat():
        """Generate a flat edge side (no spline bump)."""
        return Side(None, True, kind="flat")

    @staticmethod
    def unspecified():
        """Placeholder side for solver input (no constraint)."""
        return Side(None, True, kind="unspecified")
        
    
    
    # compare sides of pieces for placeholder use
    # returns error in [0, 1/4], so for side errors add up to [0, 1]
    def distance(self, other: "Side", polarity_check=True):
        # sides legend: __flat, --unspecified, _._normal
        
        # handle all flat cases
        # __ vs __
        # __ vs --
        # -- vs __
        # __ vs _._
        # _._ vs __
        if self.kind == "flat" or other.kind == "flat":
            return 0 if self.kind == other.kind else float('inf')
        
        # don't allow matching an underspecified piece
        # -- vs --
        # _._ vs --
        if other.kind == "unspecified":
            return float('inf')
            
        # handle case where any normal edge matches
        # -- vs _._
        if self.kind == "unspecified":
            return 0
        

        # normal kind — check polarity + geometry
        # _._ vs _._
        if polarity_check and self.male == other.male:
            return float('inf')
        if self.spline is None or other.spline is None:
            return float('inf')
        
        # squared L2 distance
        distances = np.sum((self.spline - other.spline)**2, axis=1)
        return np.mean(distances) / 4.0
                
    # symmetric, compares actual pieces sides
    def matching_cost(self, other):
        if self.male == other.male: return np.inf
        if self.kind == "flat" or other.kind == "flat": return np.inf
        if self.kind == "unspecified" or other.kind == "unspecified": return np.inf
        
        # squared L2 distance
        distances = np.sum((self.spline - other.spline)**2, axis=1)
        return np.mean(distances)
        

class Piece:
    def __init__(self, right, bottom, left, top, id=None):
        self.sides = [right, bottom, left, top]
        self.id = id
        
    def right(self, orientation):
        return self.sides[(-orientation) % 4]
        
    def bottom(self, orientation):
        return self.sides[(-orientation + 1) % 4]
        
    def left(self, orientation):
        return self.sides[(-orientation + 2) % 4]
        
    def top(self, orientation):
        return self.sides[(-orientation + 3) % 4]
        
    def invert(self): # change polarity of sides, male <-> female
        for side in self.sides:
            side.male = not side.male
        
    # calculate matching error
    def distance(self, other, orientation = 0):        
        rotated = other.sides[-orientation:] + other.sides[:-orientation]
        return sum([side.distance(other_side) for side, other_side in zip(self.sides, rotated)])
        
        
class Puzzle:
    # generate random puzzle
    def __init__(self, n, m):
        assert(n*m < 2**31 - 1) # int32 for ids
        self.size = (n, m)
        
        grid = np.zeros((n, m), dtype=np.int32)
        self.pieces = {}
        counter = 0

        for i in range(n):
            for j in range(m):
                left = Side.flat()
                top = Side.flat()
                
                # match left neighbor's right
                if j > 0:
                    left = self.pieces[grid[i][j - 1]].right(0).copy()
                    left.male = not left.male
                    
                # match top neighbor's bottom
                if i > 0:
                    top = self.pieces[grid[i - 1][j]].bottom(0).copy()
                    top.male = not top.male

                # generate new sides
                right = Side.flat()
                if j < m - 1:
                    right = Side.generate()

                bottom = Side.flat()
                if i < n - 1:
                    bottom = Side.generate()
                
                piece = Piece(right, bottom, left, top, id=counter)
                grid[i][j] = piece.id
                self.pieces[piece.id] = piece
                counter += 1

        self.solution = grid
        shuffled = grid.flatten()
        np.random.shuffle(shuffled)
        self.grid = np.reshape(shuffled, self.size)
        
    
    
# error between randomly generated, matching sides
# helps to set error thresholds for early abortion during puzzle solving
@lru_cache
def getRandomMatchError(iterations = 1000):
    errors = np.zeros(iterations)
    for i in range(iterations):
        s0 = Side.generate()
        s1 = s0.copy()
        
        s0.male = True
        s1.male = False
        
        errors[i] = s0.distance(s1)
    
    mean = np.mean(errors)
    std = np.std(errors)
    print(f'RandomMatchError: samples: {len(errors)}, mean: {mean:.5f}, std: {std:.5f}')
    return mean, std
    

# calculate errors between randomly generated splines
@lru_cache
def getRandomError(iterations = 1000):
    errors = np.zeros(iterations)
    for i in range(iterations):
        s0 = Side.generate()
        s1 = Side.generate()
        
        s0.male = True
        s1.male = False
        
        errors[i] = s0.distance(s1)
        
    mean = np.mean(errors)
    std = np.std(errors)
    print(f'RandomError: samples: {len(errors)}, mean: {mean:.5f}, std: {std:.5f}')
    return mean, std

class State:
    def __init__(self, grid, orientations, remaining, error):
        self.grid = grid
        self.orientations = orientations
        self.remaining = remaining # set of piece ids
        self.error = error
        
    # use < between states for priority queue DFS
    def __lt__(self, other):
        
        return self.error < other.error
        
        # TODO: use admissible heuristic (never overestimates) to guarantee global solution
        # use expected error
        mean, std = getRandomMatchError()
        val = 4 * mean
        e0 = len(self.remaining) * val
        e1 = len(other.remaining) * val
        return self.error + e0 < other.error + e1
    
class Solver:
    def __init__(self, puzzle: Puzzle, visualizer: Visualizer):
        self.puzzle = puzzle
        self.visualizer = visualizer
        
        n, m = puzzle.size
        
          
        def _spiral_fill_negative(n, m):
            # grid holds ids of already placed pieces, negative values indicate preferred order of placing next missin piece. more neg = place first
            # [[ -73  -74  -75  -76  -77  -78  -79  -80  -81  -82]
            #  [ -72  -43  -44  -45  -46  -47  -48  -49  -50  -83]
            #  [ -71  -42  -21  -22  -23  -24  -25  -26  -51  -84]
            #  [ -70  -41  -20   -7   -8   -9  -10  -27  -52  -85]
            #  [ -69  -40  -19   -6   -1   -2  -11  -28  -53  -86]
            #  [ -68  -39  -18   -5   -4   -3  -12  -29  -54  -87]
            #  [ -67  -38  -17  -16  -15  -14  -13  -30  -55  -88]
            #  [ -66  -37  -36  -35  -34  -33  -32  -31  -56  -89]
            #  [ -65  -64  -63  -62  -61  -60  -59  -58  -57  -90]
            #  [-100  -99  -98  -97  -96  -95  -94  -93  -92  -91]]
            grid = np.full((n, m), -1, dtype=np.int32)
            num = -1

            # Find center cell
            x, y = n // 2, m // 2
            if n % 2 == 0:
                x -= 1
            if m % 2 == 0:
                y -= 1

            # Directions: right, down, left, up
            dirs = [(0,1), (1,0), (0,-1), (-1,0)]
            grid[x, y] = num
            num -= 1

            step = 1
            while True:
                for d in range(4):  # right, down, left, up
                    dx, dy = dirs[d]
                    for _ in range(step + (d // 2)):
                        x += dx
                        y += dy
                        if 0 <= x < n and 0 <= y < m:
                            grid[x, y] = num
                            num -= 1
                        if num < -n * m:
                            return grid
                step += 2

        self.initialGrid = _spiral_fill_negative(n, m)

        # logging
        self.grid_counter = np.zeros((n, m)) # count how many trys are needed per location during solving
    
    # given linear grid index, return placeholder piece at this location (for later searching)
    def describePiece(self, grid, orientations, position):
        i, j = divmod(position, m)
        
        if j == 0:
            left = Side.flat()
        else:
            neighbor_id = grid[i][j - 1]
            if neighbor_id < 0:
                left = Side.unspecified()
            else:
                piece = self.puzzle.pieces[neighbor_id]
                left = piece.right(orientations[i][j - 1]).copy()
        
        if i == 0:
            top = Side.flat()
        else:
            neighbor_id = grid[i - 1][j]
            if neighbor_id < 0:
                top = Side.unspecified()
            else:
                piece = self.puzzle.pieces[neighbor_id]
                top = piece.bottom(orientations[i - 1][j]).copy()
        
        if j == m - 1:
            right = Side.flat()
        else:
            neighbor_id = grid[i][j + 1]
            if neighbor_id < 0:
                right = Side.unspecified()
            else:
                piece = self.puzzle.pieces[neighbor_id]
                right = piece.left(orientations[i][j + 1]).copy()
        
        if i == n - 1:
            bottom = Side.flat()
        else:
            neighbor_id = grid[i + 1][j]
            if neighbor_id < 0:
                bottom = Side.unspecified()
            else:
                piece = self.puzzle.pieces[neighbor_id]
                bottom = piece.top(orientations[i + 1][j]).copy()

        placeholder = Piece(
            right,
            bottom,
            left,
            top
        )
        
        return placeholder
    
    
    # find globally optimal solution (NP-hard)
    def branchAndBound(self, verbose = False):
        
        mean, std = getRandomMatchError()
        self.randomErrorThreshold = 4 * mean # per piece error threshold, set to 0 for perfect matching
        self.abortThreshold = n*m*self.randomErrorThreshold # solutions with this error are regarded good enough and search is aborted
              
        
        # create first state
        n, m = self.puzzle.size
        state = State(
            grid = self.initialGrid,
            orientations = np.zeros((n, m), dtype=np.int32),
            remaining = {p.id for p in self.puzzle.pieces.values()}, # set of pieces remaining to be placed
            error = 0.0, # total error is the sum of all side matching errors
            )
        
        best_error = float('inf')
        best_solution = state
        
        # logging info
        states_explored = 0
        is_local_solution = False
        
        queue = []
        heapq.heappush(queue, state)
        
        while queue:
            state = heapq.heappop(queue)
            
            # prune
            if state.error >= best_error:
                continue
                
            states_explored += 1
            
            if verbose: 
                print_grid(state.grid, states_explored)
                time.sleep(0.1)
            
            # found valid solution
            if len(state.remaining) == 0:
                best_error = state.error
                best_solution = state
                
                if verbose: print(f"\nLocal solution found, error: {state.error:.5f}")
                
                if state.error < self.abortThreshold:
                    is_local_solution = True
                    break
                    
                continue
            
            # ordered list of best matching pieces to try next
            next_candidates = np.array([(p_id, orientation) for p_id in state.remaining for orientation in [0, 1, 2, 3]])
            
            # describe missing piece
            next_position = np.argmin(state.grid)
            placeholder = self.describePiece(state.grid, state.orientations, next_position)
            
            # branch
            i, j = divmod(next_position, m)
            
            if verbose: self.grid_counter[i, j] += 1 # log progress
            
            for index, (candidate_id, orientation) in enumerate(next_candidates):
                candidate = self.puzzle.pieces[candidate_id]
                error = state.error + placeholder.distance(candidate, orientation)
                
                if error < best_error:
                    grid = state.grid.copy()
                    orientations = state.orientations.copy()
                    remaining = state.remaining.copy()
                    
                    grid[i][j] = candidate_id
                    orientations[i][j] = orientation
                    remaining.remove(candidate_id)
                    
                    new_state = State(
                        grid = grid,
                        orientations = orientations,
                        remaining = remaining,
                        error = error
                    )
                    
                    heapq.heappush(queue, new_state)
        
        if not is_local_solution and verbose: print(f"\nGlobally optimal solution found with error: {best_error:.5f}")
        
        stats = {"states_explored": states_explored, "is_local_solution": is_local_solution}
        return best_solution.grid, best_solution.orientations, stats
    
    
    def greedy(self):
        n, m = self.puzzle.size
        remaining = {p.id for p in self.puzzle.pieces.values()}
        grid = self.initialGrid
        orientations = np.zeros((n, m), dtype=np.int32)
        total_error = 0.0
        
        while remaining:
            #print_grid(grid)
            #time.sleep(0.1)
            
            start = time.perf_counter()
            
            all_candidates = np.array([(p_id, orientation) for p_id in remaining for orientation in [0, 1, 2, 3]])
            #print(len(all_candidates))
            
            # describe missing piece
            next_position = np.argmin(grid)
            placeholder = self.describePiece(grid, orientations, next_position)
            
            # branch
            i, j = divmod(next_position, m)
            
            error = np.inf
            best = None
            best_orientation = None
            for index, (candidate_id, orientation) in enumerate(all_candidates):
                candidate = self.puzzle.pieces[candidate_id]
                e = placeholder.distance(candidate, orientation)
                if e < error:
                    error = e
                    best = candidate_id
                    best_orientation = orientation


            if best is None:
                print("ERROR")
                exit()
                
            grid[i, j] = best
            orientations[i, j] = best_orientation
            remaining.remove(best)
            self.visualizer.showPuzzle(grid, orientations, self.puzzle.pieces)
            
            end = time.perf_counter()
            print(f"Took {end - start:.6f} seconds")
            
        
        print(grid)
        
        cv2.waitKey(0)        # Keep window open until key press
        cv2.destroyAllWindows()
        return grid

    
    # solves puzzle given only a subset of pieces
    def partial_solve(self):
        subset = 1.0 # 1: all pieces, 0.5: half of the pieces available
        
        n, m = self.puzzle.size
        all_pieces = list(self.puzzle.pieces.values())
        pieces = random.sample(all_pieces, int(n * m * subset))
        pieces_dict = {piece.id: piece for piece in pieces}
        
        mean, std = getRandomMatchError()
        edge_matching_threshold = mean + 2*std
        
        # store pairwise side-matching costs using priority queue
        queue = []
        
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                piece0 = pieces[i]
                piece1 = pieces[j]
                
                for orientation0, side0 in enumerate(piece0.sides):
                    for orientation1, side1 in enumerate(piece1.sides):
                        cost = side0.matching_cost(side1)
                        
                        if cost > edge_matching_threshold: continue
                        side_match = (cost, piece0.id, piece1.id, orientation0, orientation1)
                        queue.append(side_match)
        
        heapq.heapify(queue)
        
        # initialize 1-piece clusters
        clusters = {}
        for piece in pieces:
            clusters[piece.id] = Cluster(piece.id)
        
        while queue:
            # sides to combine
            _, piece0, piece1, orientation0, orientation1 = heapq.heappop(queue)
            
            cluster0 = clusters[piece0]
            cluster1 = clusters[piece1]
            
            if cluster0 is cluster1: continue # already in same cluster
            
            
            ## align clusters
            # translation
            pos0, _ = cluster0.poses[piece0]
            pos1, _ = cluster1.poses[piece1]
            if orientation0 == 0: #right
                offset = Position(1, 0)
            if orientation0 == 1: #bottom
                offset = Position(0, -1)
            if orientation0 == 2: #left
                offset = Position(-1, 0)
            if orientation0 == 3: #top
                offset = Position(0, 1)
                
            target_pos = pos0 + offset
            
            # cluster1: translate such that piece1 is at origin
            for piece, (position, orientation) in cluster1.poses.items():
                cluster1.poses[piece] = (position - pos1, orientation)
           
            # rotation (around piece1)
            # manually found by checking all 16 possibilities
            rotation_map = {(0, 0): 2,
                            (0, 1): 1,
                            (0, 2): 0,
                            (0, 3): 3,
                            (1, 0): 3,
                            (1, 1): 2,
                            (1, 2): 1,
                            (1, 3): 0,
                            (2, 0): 0,
                            (2, 1): 3,
                            (2, 2): 2,
                            (2, 3): 1,
                            (3, 0): 1,
                            (3, 1): 0,
                            (3, 2): 3,
                            (3, 3): 2
                            }
            rotation = rotation_map[(orientation0, orientation1)]
            
            # cluster1: rotate around origin
            for piece, (position, orientation) in cluster1.poses.items():
                cluster1.poses[piece] = (position.rotate(rotation), (orientation + rotation)%4)
            
            # cluster1: translate to target position
            for piece, (position, orientation) in cluster1.poses.items():
                cluster1.poses[piece] = (position + target_pos, orientation)
            
            
            ## check for compatibility
            def _compatible(cluster0, cluster1):
                for piece, (position, _) in cluster1.poses.items():
                    
                    # check if position is free
                    for _, (target_position, _) in cluster0.poses.items():
                        if position == target_position: return False
                        
                    # check neighborhood
                    for orient, neighbor in enumerate(position.neighborhood()):
                        
                        # check if neighbor exists
                        if not cluster0.exists(neighbor): continue
                        
                        piece0 = cluster0.at(neighbor)
                        side0 = pieces_dict[piece0].sides[(orient+2)%4]
                        side1 = pieces_dict[piece].sides[orient]
                        
                        # check if sides match
                        if side0.matching_cost(side1) > edge_matching_threshold:
                            return False
                        
                        orient += 1
                        side0 = pieces_dict[piece0].sides[orient%4]
                        side1 = pieces_dict[piece].sides[orient%4]
                        
                        # check for 1st border condition
                        if not((side0.kind == "flat") == (side1.kind == "flat")):
                            return False # either both flat, or none flat
                        
                        orient += 2
                        side0 = pieces_dict[piece0].sides[orient%4]
                        side1 = pieces_dict[piece].sides[orient%4]
                        
                        # check for 2nd border condition
                        if not((side0.kind == "flat") == (side1.kind == "flat")):
                            return False # either both flat, or none flat
                
                return True
             
            if not _compatible(cluster0, cluster1): continue
            
            # compatible clusters found -> merge them
            print(f'\nMerged cluster {cluster0.id} and cluster {cluster1.id}:')
            
            # remove old cluster references
            for piece in cluster0.pieces + cluster1.pieces:
                del clusters[piece]
            
            # add merged cluster
            merged = cluster0.merge(cluster1)
            for piece in merged.pieces:
                clusters[piece] = merged
            
            grid, orient = merged.grid()
            print_grid(grid, orient, None, False)
            
            # self.visualizer.showPuzzle(grid, orient, self.puzzle.pieces)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            
            
        
        def _stack_grids(grids, fill_value=-1):
            """
            Stacks 2D numpy integer arrays with 1-row and 1-column spacing between them.

            Args:
                grids (list of np.ndarray): each of shape (n_i, m_i)
                fill_value: value to fill in the spacing and empty cells

            Returns:
                stacked_grid: large grid containing all subgrids
            """
            if not grids:
                return np.array([[]], dtype=np.int32)

            # Arrange in a roughly square layout
            rows = int(np.ceil(np.sqrt(len(grids))))
            cols = int(np.ceil(len(grids) / rows))

            # Determine max grid size
            max_h = max(grid.shape[0] for grid in grids)
            max_w = max(grid.shape[1] for grid in grids)

            # Size of final grid with spacing
            total_h = rows * max_h + (rows - 1)
            total_w = cols * max_w + (cols - 1)

            # Initialize result grid
            result = np.full((total_h, total_w), fill_value, dtype=np.int32)

            for idx, grid in enumerate(grids):
                r = idx // cols
                c = idx % cols
                start_y = r * (max_h + 1)
                start_x = c * (max_w + 1)
                h, w = grid.shape
                result[start_y:start_y + h, start_x:start_x + w] = grid

            return result    


        # collect final (unique) clusters
        unique_clusters = list(set(clusters.values()))

        # collect their grids
        all_grids = []
        all_orients = []
        for cluster in unique_clusters:
            grid, orient = cluster.grid()
            all_grids.append(grid)
            all_orients.append(orient)

        # stack and display
        final_grid = _stack_grids(all_grids)
        final_orient = _stack_grids(all_orients)
        self.visualizer.showPuzzle(final_grid, final_orient, self.puzzle.pieces)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

     
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)
        
    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)
    
    def __neg__(self):
        return Position(-self.x, -self.y)
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))
        
    def __str__(self):
        return f'({self.x}, {self.y})'
        
    def __repr__(self):
        return f'({self.x}, {self.y})'
        
    def rotate(self, orientation):
        # CW rotation by orientation * 90°
        orientation = orientation % 4
        if orientation == 0: return Position(self.x, self.y)
        if orientation == 1: return Position(self.y, -self.x)
        if orientation == 2: return Position(-self.x, -self.y)
        if orientation == 3: return Position(-self.y, self.x)

    def neighborhood(self):
        # right, bottom, left, top
        return [Position(self.x + 1, self.y),
                Position(self.x, self.y - 1),
                Position(self.x - 1, self.y),
                Position(self.x, self.y + 1)
               ]


class Cluster:
    def __init__(self, piece_id):
        self.id = piece_id
        self.pieces = [piece_id]
        # position and orientation of pieces
        self.poses = {piece_id: (Position(0, 0), 0)}
        
    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
        
    def exists(self, position):
        for pos, _ in self.poses.values():
            if pos == position:
                return True
        
        return False
        
    def at(self, position):
        for key, (pos, _) in self.poses.items():
            if pos == position:
                return key
        
        raise ValueError(f"no piece at position {position}.")
        
    def merge(self, other):
        merged = Cluster(min(self.id, other.id))
        merged.pieces = self.pieces + other.pieces
        merged.poses = self.poses | other.poses
        return merged
        
    def grid(self):
        if not self.poses:
            return []

        # Extract all positions
        all_positions = [pos for pos, _ in self.poses.values()]

        # Compute bounding box
        min_x = min(p.x for p in all_positions)
        min_y = min(p.y for p in all_positions)
        max_x = max(p.x for p in all_positions)
        max_y = max(p.y for p in all_positions)

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        grid = np.full((height, width), -1, dtype=np.int32)
        orientations = np.full((height, width), -1, dtype=np.int32)

        for piece_id, (pos, orient) in self.poses.items():
            x = pos.x - min_x
            y = pos.y - min_y
            grid[height - y - 1, x] = piece_id  # row = -y, col = x
            orientations[height - y - 1, x] = orient

        return grid, orientations
        
        

# check grid agains puzzle solution (rotation invariant)
def is_valid_solution(grid, solution):
    return np.array_equal(grid, solution) \
        or np.array_equal(grid, np.rot90(solution, k=1)) \
        or np.array_equal(grid, np.rot90(solution, k=2)) \
        or np.array_equal(grid, np.rot90(solution, k=3))

def performance_measuring(n, m):
    print(f'Size: {n}x{m}')
    
    runtimes = []
    states_explored = []
    
    for i in range(50):
        puzzle = Puzzle(n, m)
    
        solver = Solver(puzzle)
        start = time.perf_counter()
        grid, orientations, stats = solver.branchAndBound()
        end = time.perf_counter()
        runtimes.append(end-start)
        #print(f"Took {end - start:.6f} seconds")
        
        assert(is_valid_solution(grid, puzzle.solution))
        
        states_explored.append(stats["states_explored"])
    
    print(f'Runtime: {np.mean(runtimes)}')
    print('States')
    print(f'Mean: {np.mean(states_explored)}, Std: {np.std(states_explored)}, Median: {np.median(states_explored)}')
    print(f'Min: {np.min(states_explored)}, Max: {np.max(states_explored)}')




if __name__ == "__main__":
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    n, m = 15,15
    #performance_measuring(n, m); exit()

    #show_distortion_grid(grid_size=(3, 3), num_points=100); exit()
    
    puzzle = Puzzle(n, m)
    vis = VisualizerCV(cell_size=32, upscale_factor=2)
    solver = Solver(puzzle, vis)
    #grid = solver.greedy();exit()
    grid = solver.partial_solve();exit()
    
    
    grid, orientations, stats = solver.branchAndBound(verbose=True)
    print(stats)
    assert(is_valid_solution(grid, puzzle.solution))
    
    
    puzzle.grid = grid
    for i in range(n):
        for j in range(m):
            piece = puzzle.pieces[puzzle.grid[i][j]]
            piece.orientation = orientations[i][j]
    