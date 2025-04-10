# Simulated Puzzle to solve

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import numpy as np

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
        self.draw_side(ax, x + cs/2, y, piece.top, 'top')
        self.draw_side(ax, x + cs, y + cs/2, piece.right, 'right')
        self.draw_side(ax, x + cs/2, y + cs, piece.bottom, 'bottom')
        self.draw_side(ax, x, y + cs/2, piece.left, 'left')
        
        # Draw piece ID in the center
        ax.text(x + cs / 2, y + cs / 2, str(piece.id),
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

            # Parameters for visual gap
            edge_margin = 0.1 * self.cell_size
            shrink = 0.2  # percentage to shrink the spline horizontally (0.0–1.0)

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


        
class Side:
    def __init__(self, spline: np.ndarray, male: bool, kind: str = "normal"):
        self.spline = spline
        self.male = male
        self.kind = kind  # "normal", "flat", or "unspecified"

    def copy(self):
        return Side(np.copy(self.spline) if self.spline is not None else None,
                    self.male,
                    self.kind)

    @staticmethod
    def generate(num_points: int, height: float):
        """Generate a normal side with a random bump."""
        x = np.linspace(0, 1, num_points)
        y = np.zeros(num_points)
        idx = np.random.randint(0, num_points)
        y[idx] = height
        spline = np.stack((x, y), axis=1)
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
    def distance(self, other: "Side"):
        if not other:
            raise ValueError("Side was None, not supported")
            
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
        
        
        # use peak distance
        return self.max_dist(other)


    # spline dist measure, distance between peaks
    def max_dist(self, other: "Side"):
        points = self.spline.shape[0]
        id1 = np.argmax(np.abs(self.spline[:, 1]))
        id2 = np.argmax(np.abs(other.spline[:, 1]))
        return np.abs(id1 - id2)
        
  

class Piece:
    def __init__(self, right, bottom, left, top, id=None):
        self.right = right
        self.bottom = bottom
        self.left = left
        self.top = top
        self.id = id
        
    def invert(self): # change polarity of sides, male <-> female
        if self.right: self.right.male = not self.right.male
        if self.bottom: self.bottom.male = not self.bottom.male
        if self.left: self.left.male = not self.left.male
        if self.top: self.top.male = not self.top.male
        
        
class Puzzle:
    def __init__(self, n, m, num_points=9, height=0.1):
        assert(n*m < 2**32) # uint32 for ids
        self.size = (n, m)
        self.grid = [[None for _ in range(m)] for _ in range(n)]
        self.pieces = []
        self.pieces_dict = {}
        self.solution = [[]]
        self.num_points = num_points
        self.height = height

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
                    left = self.pieces_dict[grid[i][j - 1]].right.copy()
                    left.male = not left.male
                # match top neighbor's bottom
                if i > 0:
                    top = self.pieces_dict[grid[i - 1][j]].bottom.copy()
                    top.male = not top.male

                # generate new sides
                right = Side.flat()
                if j < m - 1:
                    right = Side.generate(self.num_points, self.height)

                bottom = Side.flat()
                if i < n - 1:
                    bottom = Side.generate(self.num_points, self.height)

                piece = Piece(right, bottom, left, top, id=ids[counter])
                grid[i][j] = piece.id
                #self.pieces.append(piece)
                self.pieces_dict[piece.id] = piece
                counter += 1

        self.solution = grid
        #random.shuffle(self.pieces)
        shuffled = grid.flatten()
        np.random.shuffle(shuffled)
        self.grid = np.reshape(shuffled, self.size)
        #for p in self.pieces:
        #    self.pieces_dict[p.id] = p
        #self.grid = np.reshape(self.pieces, self.size).tolist()

class State:
    def __init__(self, grid, remaining, error, next):
        self.grid = grid
        self.remaining = remaining # set of piece ids
        self.next_candidates = np.array(list(remaining), dtype=np.uint32) # ordered list of best matching pieces to try
        self.error = error
        self.next = next # linear index in grid where to place next piece
        self.errors = np.zeros(len(remaining))
        
   
class Solver:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        
    # find globally optimal solution (NP-hard)
    def branchAndBound(self):
        n, m = self.puzzle.size
        # state is defined as grid of placed pieces, and list of pieces to place
        # total error is the sum of all side matching errors
        state = State(
            grid = np.zeros((n, m), dtype=np.uint32), # empty grid
            remaining = {p.id for p in self.puzzle.pieces_dict.values()}, # set of pieces remaining to be placed
            error = 0.0,
            next = 0 # indicates up to which position in grid the puzzle is filled
            )
        best_error = float('inf')
        best_solution = state
        
        queue = [state]
        
        while queue:
            state = queue.pop() # use as stack -> DFS
            
            # prune
            if state.error >= best_error:
                #print("PRUNED")
                continue
               
            # found valid solution
            if len(state.remaining) == 0:
                #print(f"Solution found, error: {state.error}")
                #print(state.grid)
                best_error = state.error
                best_solution = state
            
            # branch
            state = self.heuristic(state) # order pieces before exploring

            # use descending order to allow DFS
            state.errors = state.errors[::-1]
            state.next_candidates = state.next_candidates[::-1]
            
            # prune
            new_errors = state.errors + state.error
            mask = new_errors < best_error
            state.errors = state.errors[mask]
            state.next_candidates = state.next_candidates[mask]
            
            
            i, j = divmod(state.next, m)
            for index, candidate_id in enumerate(state.next_candidates):
                                
                grid = state.grid.copy()
                grid[i][j] = candidate_id
                remaining = state.remaining.copy()
                remaining.remove(candidate_id)
                new_state = State(
                    grid = grid,
                    remaining = remaining,
                    error = new_errors[index],
                    next = state.next + 1
                )
                
                queue.append(new_state)
        
        print(f"DONE with error: {best_error}")
        print(best_solution.grid)
        
        return best_solution.grid
        #self.puzzle.grid = best_solution.grid
    
    
    # sort remaining pieces by error
    def heuristic(self, state):
        n, m = self.puzzle.size
        # describe missing piece
        i, j = divmod(state.next, m)
        left = self.puzzle.pieces_dict[state.grid[i][j - 1]].right.copy() if j > 0 else Side.flat()
        top = self.puzzle.pieces_dict[state.grid[i - 1][j]].bottom.copy() if i > 0 else Side.flat()
        right = Side.unspecified() if j < m - 1 else Side.flat()
        bottom = Side.unspecified() if i < n - 1 else Side.flat()

        placeholder = Piece(
            right,
            bottom,
            left,
            top
        )
        
        # calculate errors
        for index, candidate_id in enumerate(state.next_candidates):
            candidate = self.puzzle.pieces_dict[candidate_id]
            state.errors[index] = (
                placeholder.left.distance(candidate.left) + 
                placeholder.right.distance(candidate.right) + 
                placeholder.top.distance(candidate.top) + 
                placeholder.bottom.distance(candidate.bottom)
            )
            
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
    
    
    # find locally optimal solution, may fail to solve puzzle
    def solve(self):
        n, m = self.puzzle.size
        grid = [[None for _ in range(m)] for _ in range(n)]

        # Pick a valid starting piece: top-left corner (no top, no left)
        for i, piece in enumerate(self.puzzle.pieces):
            if piece.top.kind == 'flat' and piece.left.kind == 'flat':
                start_piece = self.puzzle.pieces.pop(i)
                #print(start_piece.id)
                break
        else:
            raise ValueError("No valid starting piece found")

        grid[0][0] = start_piece

        for i in range(n):
            for j in range(m):
                if grid[i][j]:
                    continue
                
                left = grid[i][j - 1].right.copy() if j > 0 else Side.flat()
                top = grid[i - 1][j].bottom.copy() if i > 0 else Side.flat()
                
                right = Side.unspecified() if j < m - 1 else Side.flat()
                bottom = Side.unspecified() if i < n - 1 else Side.flat()

                placeholder = Piece(
                    right,
                    bottom,
                    left,
                    top
                )
                
                match, err = self.findBestMatch(placeholder)
                print(f'piece {match.id} matched at {i, j} with error: {err}')
                grid[i][j] = match
                
        self.puzzle.grid = grid
    
    # find best matching piece
    def findBestMatch(self, piece):
        best_idx = None
        min_error = float('inf')

        for i, candidate in enumerate(self.puzzle.pieces):
            total_error = \
                piece.left.distance(candidate.left) + \
                piece.right.distance(candidate.right) + \
                piece.top.distance(candidate.top) + \
                piece.bottom.distance(andidate.bottom)
                
            if total_error < min_error:
                best_idx = i
                min_error = total_error
                
        if best_idx is None or total_error == float('inf'):
            raise ValueError("No suitable match found")
        
        return self.puzzle.pieces.pop(best_idx), total_error
        
        



if __name__ == "__main__":
    
    random.seed(42)
    np.random.seed(42)
    
   
    vis = Visualizer()
    
    
    n, m = 6,6
    puzzle = Puzzle(n, m)
    puzzle.generate()
    
    #vis.showPuzzle(puzzle)

    solver = Solver(puzzle)
    #solver.solve()
    puzzle.grid = solver.branchAndBound()
    
    # check solution
    for i in range(n):
        for j in range(m):
            assert(puzzle.grid[i, j] == puzzle.solution[i, j])
    
    print('Solution valid')
    vis.showPuzzle(puzzle)