# Jigsaw Puzzle Solver

Takes photos of your puzzle pieces and provides you with the correct matches. Uses contour detection of pieces to find corresponding matching sides.

Technical steps:

- Input: photo of multiple puzzle pieces on white background, back side of the puzzle pieces is showing.
- Extract contour of all pieces
- Extract 4 corners from countour
- Extract 4 sides as splines
- Compare sides using l2 norm to find best matches
