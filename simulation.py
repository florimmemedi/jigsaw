# Simulated Puzzle to solve

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import math
import heapq
import time
import os
from functools import lru_cache

import numpy as np
import scipy

def print_grid(grid):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
    for row in grid:
        print(' '.join(f'{cell:3}' if cell >= 0 else '  .' for cell in row))

class Visualizer:
    def __init__(self, cell_size=1):
        self.cell_size = cell_size

    def showPuzzle(self, puzzle):
        n, m = puzzle.size
        scale = 2  # inches per cell
        fig, ax = plt.subplots(figsize=(m * scale, n * scale))
        ax.set_aspect('equal')
        ax.set_xlim(0, m * self.cell_size)
        ax.set_ylim(0, n * self.cell_size)
        ax.invert_yaxis()  # top-left origin
        ax.axis('off')

        for i in range(n):
            for j in range(m):
                piece_id = puzzle.grid[i][j]
                self.draw_piece(ax, i, j, puzzle.pieces_dict[piece_id])

        plt.show()

    def draw_piece(self, ax, row, col, piece):
        #if piece.id != 1: return
        #print(piece.right)
        
        x = col * self.cell_size
        y = row * self.cell_size
        cs = self.cell_size

        # Draw the square for the piece
        rect = patches.Rectangle((x, y), cs, cs, linewidth=1, edgecolor='grey', facecolor='lightgrey')
        ax.add_patch(rect)

        # Draw side indicators (text or small lines)
        self.draw_side(ax, x + cs/2, y, piece.top(), 'top')
        self.draw_side(ax, x + cs, y + cs/2, piece.right(), 'right')
        self.draw_side(ax, x + cs/2, y + cs, piece.bottom(), 'bottom')
        self.draw_side(ax, x, y + cs/2, piece.left(), 'left')
        
        # Draw piece ID in the center
        show_orientation = False
        text = f'{str(piece.id)} ({str(piece.orientation)})' if show_orientation else f'{str(piece.id)}'
        ax.text(x + cs / 2, y + cs / 2, text, ha='center', va='center', fontsize=8, color='black')

    def draw_side(self, ax, x_data, y_data, side, position):
        if not side:
            return
            
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
        delta_e = 0.02
        projective = {
            "scale_x": 1.0,
            "shear_x": np.random.uniform(-delta_e, delta_e),
            "translate_x": 0.0,
            "shear_y": np.random.uniform(-delta_e, delta_e),
            "scale_y": 1.0,
            "translate_y": 0.0,
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
        self.orientation = 0
        self.id = id
        
    def right(self):
        return self.sides[(self.orientation + 0) % 4]
        
    def bottom(self):
        return self.sides[(self.orientation + 1) % 4]
        
    def left(self):
        return self.sides[(self.orientation + 2) % 4]
        
    def top(self):
        return self.sides[(self.orientation + 3) % 4]
        
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
        
        grid = np.zeros((n, m), dtype=np.uint32)
        self.pieces_dict = {}
        # random ids for pieces
        ids = np.arange(n*m)
        np.random.shuffle(ids)
        counter = 0

        for i in range(n):
            for j in range(m):
                left = Side.flat()
                top = Side.flat()
                
                # match left neighbor's right
                if j > 0:
                    left = self.pieces_dict[grid[i][j - 1]].right().copy()
                    left.male = not left.male
                    
                # match top neighbor's bottom
                if i > 0:
                    top = self.pieces_dict[grid[i - 1][j]].bottom().copy()
                    top.male = not top.male

                # generate new sides
                right = Side.flat()
                if j < m - 1:
                    right = Side.generate()

                bottom = Side.flat()
                if i < n - 1:
                    bottom = Side.generate()
                
                piece = Piece(right, bottom, left, top, id=ids[counter])
                grid[i][j] = piece.id
                self.pieces_dict[piece.id] = piece
                counter += 1

        self.solution = grid
        shuffled = grid.flatten()
        np.random.shuffle(shuffled)
        self.grid = np.reshape(shuffled, self.size)
        
        # random pieces orientation
        for piece in self.pieces_dict.values():
           piece.orientation = np.random.randint(0, 4)
    
    
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
    def __init__(self, grid, grid_orientations, remaining, error, next):
        self.grid = grid
        self.grid_orientations = grid_orientations
        self.remaining = remaining # set of piece ids
        self.next_candidates = np.array([(p_id, orientation) for p_id in remaining for orientation in [0, 1, 2, 3]]) # ordered list of best matching pieces to try
        self.error = error
        self.next = next # linear index in grid where to place next piece
        
    # use < between states for priority queue DFS
    def __lt__(self, other):
        # TODO: use admissible heuristic (never overestimates) to guarantee global solution
        
        # use expected error
        mean, std = getRandomMatchError()
        val = 4 * mean
        e0 = len(self.remaining) * val
        e1 = len(other.remaining) * val
        return self.error + e0 < other.error + e1
    
class Solver:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
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
    def describePiece(self, state, position):
        i, j = divmod(position, m)
        
        if j == 0:
            left = Side.flat()
        else:
            neighbor_id = state.grid[i][j - 1]
            if neighbor_id < 0:
                left = Side.unspecified()
            else:
                piece = self.puzzle.pieces_dict[neighbor_id]
                piece.orientation = state.grid_orientations[i][j - 1]
                left = piece.right().copy()
        
        if i == 0:
            top = Side.flat()
        else:
            neighbor_id = state.grid[i - 1][j]
            if neighbor_id < 0:
                top = Side.unspecified()
            else:
                piece = self.puzzle.pieces_dict[neighbor_id]
                piece.orientation = state.grid_orientations[i - 1][j]
                top = piece.bottom().copy()
        
        if j == m - 1:
            right = Side.flat()
        else:
            neighbor_id = state.grid[i][j + 1]
            if neighbor_id < 0:
                right = Side.unspecified()
            else:
                piece = self.puzzle.pieces_dict[neighbor_id]
                piece.orientation = state.grid_orientations[i][j + 1]
                right = piece.left().copy()
        
        if i == n - 1:
            bottom = Side.flat()
        else:
            neighbor_id = state.grid[i + 1][j]
            if neighbor_id < 0:
                bottom = Side.unspecified()
            else:
                piece = self.puzzle.pieces_dict[neighbor_id]
                piece.orientation = state.grid_orientations[i + 1][j]
                bottom = piece.top().copy()

        placeholder = Piece(
            right,
            bottom,
            left,
            top
        )
        
        return placeholder
    
    
    # find globally optimal solution (NP-hard)
    def branchAndBound(self, verbose = False):
        n, m = self.puzzle.size
        
        state = State(
            grid = self.initialGrid, # empty grid is negative
            grid_orientations = np.zeros((n, m), dtype=np.uint32), # empty
            remaining = {p.id for p in self.puzzle.pieces_dict.values()}, # set of pieces remaining to be placed
            error = 0.0, # total error is the sum of all side matching errors
            next = np.argmin(self.initialGrid), # start with most negative position
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
            
            if verbose:
                print_grid(state.grid)
                time.sleep(0.2)
            
            states_explored += 1
            if verbose: print(f'\rExplored: {states_explored}', end='')
            
            
            # prune
            if state.error >= best_error:
                continue
               
            # found valid solution
            if len(state.remaining) == 0:
                best_error = state.error
                best_solution = state
                
                if verbose: print(f"\nLocal solution found, error: {state.error:.5f}")
                if state.error < self.abortThreshold:
                    is_local_solution = True
                    break
                    
                continue
            
            # log progress
            i, j = divmod(state.next, m)
            self.grid_counter[i, j] += 1
            
            
            
            # describe missing piece
            placeholder = self.describePiece(state, state.next)
            
            # calculate errors
            next_candidate_errors = np.zeros(len(state.next_candidates))
            for index, (candidate_id, orientation) in enumerate(state.next_candidates):
                candidate = self.puzzle.pieces_dict[candidate_id]
                next_candidate_errors[index] = placeholder.distance(candidate, orientation)
            
            # prune inf errors
            mask = next_candidate_errors != np.inf
            next_candidate_errors = next_candidate_errors[mask]
            state.next_candidates = state.next_candidates[mask]
            
            # prune worse paths
            new_errors = next_candidate_errors + state.error
            mask = new_errors < best_error
            next_candidate_errors = next_candidate_errors[mask]
            state.next_candidates = state.next_candidates[mask]
            
            # branch
            i, j = divmod(state.next, m)
            for index, (candidate_id, orientation) in enumerate(state.next_candidates):
                                
                grid = state.grid.copy()
                grid[i][j] = candidate_id
                grid_orientations = state.grid_orientations.copy()
                grid_orientations[i][j] = orientation
                remaining = state.remaining.copy()
                remaining.remove(candidate_id)
                
                new_state = State(
                    grid = grid,
                    grid_orientations = grid_orientations,
                    remaining = remaining,
                    error = new_errors[index],
                    next = np.argmin(grid) # best position to solve next
                )
                
                heapq.heappush(queue, new_state)
        
        
        if not is_local_solution:
            if verbose: print(f"\nGlobally optimal solution found with error: {best_error:.5f}")
        
        stats = {"states_explored": states_explored, "is_local_solution": is_local_solution}
        return best_solution.grid, best_solution.grid_orientations, stats
    
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
        #print(i)
        puzzle = Puzzle(n, m)
    
        solver = Solver(puzzle)
        start = time.perf_counter()
        grid, orientations, stats = solver.branchAndBound()
        end = time.perf_counter()
        runtimes.append(end-start)
        #print(f"Took {end - start:.6f} seconds")
        
        assert(is_valid_solution(grid, puzzle.solution))
        
        states_explored.append(stats["states_explored"])
    
    #print(runtimes)
    print(f'Runtime: {np.mean(runtimes)}')
    print('States')
    print(f'Mean: {np.mean(states_explored)}, Std: {np.std(states_explored)}, Median: {np.median(states_explored)}')
    print(f'Min: {np.min(states_explored)}, Max: {np.max(states_explored)}')




if __name__ == "__main__":
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    n, m = 5, 5
    #performance_measuring(n, m); exit()

    
    puzzle = Puzzle(n, m)
    solver = Solver(puzzle)
    grid, orientations, stats = solver.branchAndBound()
    print(stats)
    
    puzzle.grid = grid
    for i in range(n):
        for j in range(m):
            piece = puzzle.pieces_dict[puzzle.grid[i][j]]
            piece.orientation = orientations[i][j]
    
    vis = Visualizer()
    #vis.showPuzzle(puzzle)