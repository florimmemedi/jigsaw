# Simulated Puzzle to solve

import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.ion()  # turn on interactive mode
import cv2

import random
import math
import heapq
import time
import os
from functools import lru_cache

import numpy as np
import scipy

def print_grid(grid, states_explored = None):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
    if states_explored is not None: print(f'\rExplored: {states_explored}')
    for row in grid:
        print(' '.join(f'{cell:3}' if cell >= 0 else '  .' for cell in row))
        
        
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
            self.drawn_cells.clear()

        for i in range(n):
            for j in range(m):
                piece_id = grid[i, j]
                if piece_id < 0:
                    continue
                if (i, j) in self.drawn_cells:
                    continue
                self.draw_piece(i, j, pieces[piece_id], orientations[i, j])
                self.drawn_cells.add((i, j))

        # Downscale for display
        small = cv2.resize(self.canvas, (m * self.cell_size, n * self.cell_size), interpolation=cv2.INTER_AREA)
        cv2.imshow("Puzzle", small)
        cv2.waitKey(1)

    def draw_piece(self, row, col, piece, orientation):
        cs = self.cell_size * self.upscale
        x = col * cs
        y = row * cs

        # Draw square
        cv2.rectangle(self.canvas, (x, y), (x + cs, y + cs), (200, 200, 200), thickness=-1)
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
       
class Side:
    def __init__(self, spline: np.ndarray, male: bool, kind: str = "normal"):
        self.spline = spline
        self.male = male
        self.kind = kind  # "normal", "flat", or "unspecified"

    # apply slight deformation
    def copy(self):
        if self.spline is None:
            return Side(None, self.male, self.kind)
            
        # small perturbations
        delta_e = 0.0
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
        spline = apply_projective(np.copy(self.spline), projective)
        
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
        
    
    
    # compare sides of pieces
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
def getRandomError(iterations):
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
        
        mean, std = getRandomMatchError()
        self.randomErrorThreshold = 4 * mean # per piece error threshold, set to 0 for perfect matching
        self.abortThreshold = n*m*self.randomErrorThreshold # solutions with this error are regarded good enough and search is aborted
                
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
    
    n, m = 10, 10
    #performance_measuring(n, m); exit()

    
    puzzle = Puzzle(n, m)
    vis = VisualizerCV(cell_size=64, upscale_factor=2)
    solver = Solver(puzzle, vis)
    grid = solver.greedy();exit()
    
    grid, orientations, stats = solver.branchAndBound(verbose=True)
    print(stats)
    assert(is_valid_solution(grid, puzzle.solution))
    
    
    puzzle.grid = grid
    for i in range(n):
        for j in range(m):
            piece = puzzle.pieces[puzzle.grid[i][j]]
            piece.orientation = orientations[i][j]
    