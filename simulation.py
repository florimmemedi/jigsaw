# Simulated Puzzle to solve


import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import math
import numpy as np
import scipy

class Visualizer:
    def __init__(self, cell_size=1):
        self.cell_size = cell_size

    def showPuzzle(self, puzzle):
        n, m = puzzle.size
        fig, ax = plt.subplots()
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
        ax.text(x + cs / 2, y + cs / 2, f'{str(piece.id)} ({str(piece.orientation)})',
                ha='center', va='center', fontsize=8, color='black')

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
            "height": 0.376,
            "neck_width": 0.077,
            "neck_heigth": 0.174,
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
    def distance(self, other: "Side"):
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
        if self.male == other.male:
            return float('inf')
        if self.spline is None or other.spline is None:
            return float('inf')
        
        # L2 distance
        distances = np.linalg.norm(self.spline - other.spline, axis=1)
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
    def __init__(self, n, m):
        assert(n*m < 2**32) # uint32 for ids
        self.size = (n, m)
        self.grid = [[None for _ in range(m)] for _ in range(n)]
        self.pieces_dict = {}
        self.solution = [[]]

    def generate(self):
        n, m = self.size
        grid = np.zeros((n, m), dtype=np.uint32)
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
        p0 = self.pieces_dict[0]
        p0.orientation = 1
        #for piece in self.pieces_dict.values():
         #   piece.orientation = np.random.randint(0, 4)
        
    # calculate mean error between random pieces
    def getRandomError(self, iterations):
        n, m = self.size
        idxs = np.random.randint(0, n*m, 2*iterations)
        errors = []
        for i in range(iterations):
            piece1 = self.pieces_dict[idxs[2*i]]
            piece2 = self.pieces_dict[idxs[2*i + 1]]
            
            error = piece1.distance(piece2)
            if error != float('inf'):
                errors.append(error)
                
        print(f'samples: {len(errors)}, mean: {np.mean(errors)}, std: {np.std(errors)}')

class State:
    def __init__(self, grid, grid_orientations, remaining, error, next):
        self.grid = grid
        self.grid_orientations = grid_orientations
        self.remaining = remaining # set of piece ids
        self.next_candidates = np.array([(p_id, orientation) for p_id in remaining for orientation in [0, 1, 2, 3]]) # ordered list of best matching pieces to try
        self.error = error
        self.next = next # linear index in grid where to place next piece
        self.errors = np.zeros(len(self.next_candidates))
        
   
class Solver:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        n, m = puzzle.size
        self.randomErrorThreshold = 0.1 # per piece error threshold, set to 0 for perfect matching, gather sensible values from puzzle.getRandomError()
        self.abortThreshold = n*m*self.randomErrorThreshold # solutions with this error are regarded optimal and search aborted
        
    # find globally optimal solution (NP-hard)
    def branchAndBound(self):
        n, m = self.puzzle.size
        # state is defined as grid of placed pieces, and list of pieces still to be placed
        # total error is the sum of all side matching errors
        state = State(
            grid = np.zeros((n, m), dtype=np.uint32), # empty grid
            grid_orientations = np.zeros((n, m), dtype=np.uint32), # empty
            remaining = {p.id for p in self.puzzle.pieces_dict.values()}, # set of pieces remaining to be placed
            error = 0.0,
            next = 0, # indicates up to which position in grid the puzzle is filled
            )
        best_error = float('inf')
        best_solution = state
        
        # logging info
        states_explored = 0
        
        queue = [state]
        
        while queue:
            state = queue.pop() # use as stack -> DFS
            print(f'Explored: {states_explored}')
            
            # prune
            if state.error >= best_error:
                level = state.next
                continue
               
            # found valid solution
            if len(state.remaining) == 0:
                print(f"Local solution found, error: {state.error}")
                if state.error < self.abortThreshold:
                    return state.grid, state.grid_orientations
                    
                best_error = state.error
                best_solution = state
                continue
            
            # branch
            state = self.heuristic(state) # order pieces before exploring

            # use descending order to allow DFS
            state.errors = state.errors[::-1]
            state.next_candidates = state.next_candidates[::-1]
            
            # prune
            new_errors = state.errors + state.error
            mask = new_errors < best_error
            level = state.next
            state.errors = state.errors[mask]
            state.next_candidates = state.next_candidates[mask]
            
            
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
                    next = state.next + 1
                )
                states_explored += 1
                queue.append(new_state)
        
        print(f"Globally optimal solution found with error: {best_error}")
        #print(best_solution.grid)
        
        return best_solution.grid, best_solution.grid_orientations
    
    
    # sort remaining pieces by error
    def heuristic(self, state):
        n, m = self.puzzle.size
        
        # describe missing piece
        i, j = divmod(state.next, m)
        self.puzzle.pieces_dict[state.grid[i][j - 1]].orientation = state.grid_orientations[i][j - 1]
        left = self.puzzle.pieces_dict[state.grid[i][j - 1]].right().copy() if j > 0 else Side.flat()
        self.puzzle.pieces_dict[state.grid[i - 1][j]].orientation = state.grid_orientations[i - 1][j]
        top = self.puzzle.pieces_dict[state.grid[i - 1][j]].bottom().copy() if i > 0 else Side.flat()
        right = Side.unspecified() if j < m - 1 else Side.flat()
        bottom = Side.unspecified() if i < n - 1 else Side.flat()

        placeholder = Piece(
            right,
            bottom,
            left,
            top
        )
        
        # calculate errors
        for index, (candidate_id, orientation) in enumerate(state.next_candidates):
            candidate = self.puzzle.pieces_dict[candidate_id]
            state.errors[index] = placeholder.distance(candidate, orientation)
            
        # prune using error threshold
        mask = state.errors <= self.randomErrorThreshold
        state.errors = state.errors[mask]
        state.next_candidates = state.next_candidates[mask]
            
            
        # prune inf errors
        mask = state.errors != np.inf
        state.errors = state.errors[mask]
        state.next_candidates = state.next_candidates[mask]
        
        # sort according to error
        idx = np.argsort(state.errors)
        state.errors = state.errors[idx]
        state.next_candidates = state.next_candidates[idx]
        #print(state.errors, state.next_candidates, state.remaining)
        
        
        return state
    



if __name__ == "__main__":
    
    random.seed(42)
    np.random.seed(42)
    
   
    vis = Visualizer()
    
    
    n, m = 5, 5
    puzzle = Puzzle(n, m)
    puzzle.generate()
    
 
    #print(puzzle.getRandomError(10000))
    #exit()
    
    #vis.showPuzzle(puzzle)

    solver = Solver(puzzle)
    grid, orientations = solver.branchAndBound()
    
    # print(grid)
    # print(orientations)
    # print(puzzle.solution)
    
    # check solution (rotation invariant)
    if np.array_equal(grid, puzzle.solution) \
        or np.array_equal(grid, np.rot90(puzzle.solution, k=1)) \
        or np.array_equal(grid, np.rot90(puzzle.solution, k=2)) \
        or np.array_equal(grid, np.rot90(puzzle.solution, k=3)):
        
        print('Solution valid')
    
    
    puzzle.grid = grid
    for i in range(n):
        for j in range(m):
            piece = puzzle.pieces_dict[puzzle.grid[i][j]]
            piece.orientation = orientations[i][j]
    
    vis.showPuzzle(puzzle)