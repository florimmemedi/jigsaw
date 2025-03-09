import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import math
import pickle

import scipy as sc


# given sequence of 2d points and number of interpolation steps, return equidistantly interpolated spline
# returning "steps"-number of points (including original end-points), requires steps >= 2
def interpolateSpline(spline, steps):
    
    if steps < 2: raise Exception('Need at least two substeps for spline interpolation.')
    l = len(spline)
    if l < 2: raise Exception('Spline must have at least 2 points')
    
    #splineLength = cv2.arcLength(spline, False)
    
    diffs = np.diff(spline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dists = dists.tolist()
    dists.insert(0, 0)
    dists = np.asarray(dists)
    splineLength = np.sum(dists)
    dists = np.cumsum(dists)
    
    locations = np.linspace(0, splineLength, num=steps)
    
    xs = np.interp(locations, dists, spline[:, 0])
    ys = np.interp(locations, dists, spline[:, 1])
    
    out = np.concatenate(([xs], [ys]), axis=0).T
    #print(out)
    #exit()
    
    """
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.scatter(xs, ys)
    plt.show()
    exit()
    """
    return out

# compare 2 interpolated splines and return distance measure between them
def splineDist(s0, s1):
    if len(s0) != len(s1): raise Exception('Splines must have same resolution to be comparable.')
    
    
    acc = 0
    for p0 in s0:
        min_dist = float('inf')
        for p1 in s1:
            dist = np.linalg.norm(p0-p1)
            if dist < min_dist:
                min_dist = dist
                
        acc += min_dist
    return acc
        
        
        
        
    diffs = s0 - s1
    dists = np.linalg.norm(diffs, axis=1)
    
    """
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.scatter(s0[:, 0], s0[:, 1], c='b', label='query')
    plt.scatter(s1[:, 0], s1[:, 1], c='g', label='compare')
    plt.legend()
    plt.show()
    exit()
    """
    
    total = np.sum(dists)
    return total


# returns intersection point (x, y) of L1 = (p1, p2) and L2 = (p3, p4) or raises Exception if none found
def lineLineIntersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0: raise Exception('lines do not intersect!')
    
    x = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
    y = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
    
    return x, y

def triangleArea(s1, s2, s3):
    s = (s1 + s2 + s3) / 2
    return np.sqrt(s*(s-s1)*(s-s2)*(s-s3))
    

# returns score of how good the corners represent the puzzle piece.
# area spanned and right angle measure of corners
# assumes points are ordered counterclockwise
def corner_score(c1, c2, c3, c4, img):
    # side lengths
    s12 = np.linalg.norm(c1-c2)
    s23 = np.linalg.norm(c2-c3)
    s31 = np.linalg.norm(c3-c1)
    s34 = np.linalg.norm(c3-c4)
    s41 = np.linalg.norm(c4-c1)
    
    a1 = triangleArea(s12, s23, s31)
    a2 = triangleArea(s31, s34, s41)
    
    maxarea = img.shape[0] * img.shape[1]
    # % of area covered
    area = (a1 + a2) / maxarea
    
    # consider right angle score
    anglescore = 4
    anglescore -= abs(np.dot(c1-c2, c2-c3) / (s12 * s23))
    anglescore -= abs(np.dot(c2-c3, c3-c4) / (s23 * s34))
    anglescore -= abs(np.dot(c3-c4, c4-c1) / (s34 * s41))
    anglescore -= abs(np.dot(c4-c1, c1-c2) / (s41 * s12))
    anglescore /= 4
    #print('anglescore / area: ', anglescore, area)
    
    
    # consider ratio of side lengths
    sideRatioScore = 4
    meanSideLength = np.mean([s12, s23, s34, s41])
    s12 /= meanSideLength
    s23 /= meanSideLength
    s34 /= meanSideLength
    s41 /= meanSideLength
    sideRatioScore -= s12 / s23
    sideRatioScore -= s23 / s34
    sideRatioScore -= s34 / s41
    sideRatioScore -= s41 / s12
    
    # TODO: for now works with 1, else we get contours, where 2 points lie very close to each other!!
    #ratio = 0.5
    #return ratio*area + (1-ratio)*anglescore

    return area + anglescore + sideRatioScore


def contour2piece(contour, img):

    cs = contour
    center = np.mean(cs, axis=0)

    xs = cs[:, 0] - center[0]
    ys = cs[:, 1] - center[1]

    rs = np.sqrt(xs**2 + ys**2)

    rs = sc.signal.savgol_filter(rs, 25, 2) # window size, polynomial order

    maxima, _ = sc.signal.find_peaks(rs)
    # extend with first and last item to account for wrap around
    maxima = maxima.tolist()
    maxima.insert(0, 0)
    maxima.append(len(rs)-1)
    maxima = np.asarray(maxima)

    #print(maxima)

    #plt.plot(range(len(rs)), rs)
    #plt.scatter(maxima, rs[maxima], c='r')
    #plt.show()

    cornerCandidates = cs[maxima]
    #print(cornerCandidates)

    #plt.imshow(img)
    #plt.scatter(cornerCandidates[:, 0], cornerCandidates[:, 1])
    #plt.show()

    
    # prune corners
    n = len(cornerCandidates)
    max_score = 0
    best_corners = []
    best_corners_inds = []
    for i1 in range(n):
        for i2 in range(i1+1, n):
            for i3 in range(i2+1, n):
                for i4 in range(i3+1, n):
                    score = corner_score(cornerCandidates[i1], cornerCandidates[i2], cornerCandidates[i3], cornerCandidates[i4], img)
                    if score > max_score:
                        max_score = score
                        best_corners = np.asarray([cornerCandidates[i1], cornerCandidates[i2], cornerCandidates[i3], cornerCandidates[i4]])
                        best_corners_inds = np.asarray([i1, i2, i3, i4])

    #print(best_corners)
    best_corners_inds = maxima[best_corners_inds]
    #print(best_corners_inds)
    plt.scatter(best_corners[:, 0], best_corners[:, 1], s=10, facecolors='none', edgecolors='g')

    # get centroid
    centroid = np.mean(best_corners, axis=0)
    #print(centroid)


    #print(edge_points)
    edge_points = cs

    contours = []
    contours.append(cs[best_corners_inds[0]:best_corners_inds[1]])
    contours.append(cs[best_corners_inds[1]:best_corners_inds[2]])
    contours.append(cs[best_corners_inds[2]:best_corners_inds[3]])
    contours.append(cs.take(range(best_corners_inds[3],len(cs) + best_corners_inds[0]), mode='wrap', axis=0))

    #print(contours[3])


    """
    # refine corners
    refined_corners = []
    for corner in best_corners:
        max_dist = 0
        best_corner = corner
        for point in edge_points:
            # only consider 10 pixel radius
            if np.linalg.norm(corner-point) < 10:
                dist = np.linalg.norm(centroid-point)
                if dist > max_dist:
                    max_dist = dist
                    best_corner = point
         
        refined_corners.append(best_corner)

    refined_corners = sort_counterclockwise(np.asarray(refined_corners))

    print(refined_corners)


    # distance between point p0 to line (p1, p2)
    def pointLineDist(p1, p2, p0):
        return np.abs((p2[0]-p1[0])*(p1[1]-p0[1]) - (p1[0]-p0[0])*(p2[1]-p1[1])) / np.linalg.norm(p2-p1)

    # extract 4 sides
    lines = np.asarray([
                [refined_corners[0], refined_corners[1]],
                [refined_corners[1], refined_corners[2]],
                [refined_corners[2], refined_corners[3]],
                [refined_corners[3], refined_corners[0]]
            ])
    # assign points to closest line, or to remaining
    contours = [[], [], [], [], []]
    maxDist = np.amin(np.linalg.norm(lines[:, 0] - lines[:, 1])) * 0.1 # maximum distance from line
    for point in edge_points:
        bestDist = float('inf')
        bestContourInd = 4 # all unclassified go here
        for i, (p1, p2) in enumerate(lines):
            dist = pointLineDist(p1, p2, point)
            if dist > maxDist: continue
            if dist < bestDist:
                bestDist = dist
                bestContourInd = i
        contours[bestContourInd].append(point)
        
    unclassified = contours[4]
    contours = contours[:4]

    for point in unclassified:
        bestDist = float('inf')
        bestContourInd = 0
        for i, contour in enumerate(contours):
            for p in contour:
                dist = np.linalg.norm(point-p)
                if dist < bestDist:
                    bestDist = dist
                    bestContourInd = i
                    
        contours[bestContourInd].append(point)
    """
    
    # rotates 2D points (2 x n) about theta around the origin
    def rotatePoints(points, theta):
        R = np.asarray([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        return np.matmul(R, points) # rotate

    # normalize puzzle side, such that reference points p1, p2 lie on (-1, 0) an (1, 0) respectively
    # female contours gets rotated 180° to be comparable to males
    def normalizeSide(contour, p1, p2):
        side = contour
        side = side - (p1 + (p2-p1)/2) # center
        theta = -np.arctan2((p1[1]-p2[1]), (p1[0]-p2[0])) # rotation angle
        side = rotatePoints(side.T, theta).T
        
        denom = (abs(np.amax(side[:, 0]) - np.amin(side[:, 0])))*2
        
        if denom == 0:
            print('division by 0 error')
        side = side / (abs(np.amax(side[:, 0]) - np.amin(side[:, 0])))*2 # stretch to [-1, 1] on x-axis
        
        #plt.clf()
        #plt.scatter(side[:,0], side[:,1])
        #plt.show()
        
        
        male = True
        # check if male/female side        
        if np.amin(side[:, 1]) < -0.25:
            # female -> rotate 180°
            male = False
            side = rotatePoints(side.T, np.pi).T
        
        
            
        return side, male



    sides = []
    for i in range(len(contours)):
        contours[i] = np.asarray(contours[i])
        #print(contours[i][-1])
        #p1, p2 = lines[i]
        """
        plt.scatter(contours[0][:, 0], contours[0][:, 1], c='r', s=1)
        plt.scatter(contours[1][:, 0], contours[1][:, 1], c='g', s=1)
        plt.scatter(contours[2][:, 0], contours[2][:, 1], c='b', s=1)
        plt.scatter(contours[3][:, 0], contours[3][:, 1], c='k', s=1)
        plt.show()
        """
        sides.append(normalizeSide(contours[i], contours[i][0], contours[i][-1]))
        
    #print(contours)
    
    
    
    

    #def side2Vec(side):

    # input: one side of puzzle piece
    # output: 1 x n feature of side shape
    def side2feature(side):
        n = 100
        #feature = np.zeros((n, n))
        max_height = np.zeros((n))
        side = (side + 1) * ((n-1)/2)
        for x, y in side:
            x_ind = min(math.floor(x), n-1)
            max_height[x_ind] = max(max_height[x_ind], y)
            #feature[math.floor(y), math.floor(x)] += 1
        
        return max_height
        #return np.max(feature, axis=0) / len(side)
        
        
    
        
        
    piece = []
    for side, male in sides:
        #piece.append(side2feature(side))
        piece.append((interpolateSpline(side, 100), male))
    #piece = np.asarray(piece)
    #print(piece)
    
    


    """

    #feature0 = side2feature(sides[0])
    #print(feature0)
    #print(np.linalg.norm(side2feature(sides[0]) - side2feature(sides[3])))
    #plt.imshow(feature0)
    plt.show()
    exit()

    #plt.imshow(thresh)
    plt.gca().set_aspect('equal')
    plt.scatter(sides[2][:, 0], sides[2][:, 1], c=range(len(sides[2])))
    #plt.scatter(contours[2][:, 0], contours[2][:, 1], c='g')
    #plt.scatter(p1[1], p1[0], c='k')
    #plt.scatter(p2[1], p2[0], c='b')
    plt.show()
    exit()
    """


    """
    plt.imshow(thresh)

    # show corners
    #plt.scatter(cornerCandidates[:, 0], cornerCandidates[:, 1], c='r')
    plt.scatter(best_corners[:, 0], best_corners[:, 1], facecolors='none',edgecolors='b')
    #plt.scatter(refined_corners[:, 0], refined_corners[:, 1], c='g')
    #plt.scatter(cs[:, 0], cs[:, 1], c='k')
    #plt.scatter(intersections[:, 0], intersections[:, 1], c='k')

    # display contours
    plt.scatter(contours[0][:, 0], contours[0][:, 1], c='r')
    plt.scatter(contours[1][:, 0], contours[1][:, 1], c='g')
    plt.scatter(contours[2][:, 0], contours[2][:, 1], c='b')
    plt.scatter(contours[3][:, 0], contours[3][:, 1], c='k')

    plt.scatter(centroid[1], centroid[0], c='k')
    plt.show()
    """

    return piece, sides, contours

def prepImage(img):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.medianBlur(gray, ksize=5)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.blur(thresh, ksize=(3, 3))
    return thresh

def img2pieces(path, count_offset, img_name):
    img = cv2.imread(path)
    #print(np.shape(img))

    thresh = prepImage(img)
    plt.clf()
    plt.imshow(img)
    #plt.show()

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # select correct contours
    pieces = []
    count = count_offset
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i]) 
        
        # minimum size
        if w < 20 or h < 20: continue
        
        # maximum size
        bound = 0.75
        if w > img.shape[1] * bound or h > img.shape[0] * bound: continue
        
        area = cv2.contourArea(contours[i])
        if area < 20000: continue
        #if area > 100000: continue
        
        cs = np.squeeze(np.asarray(contours[i]), axis=1)
        
        center_x, center_y = np.mean(cs, axis=0)
        text = plt.text(center_x, center_y, f'{count}', ha='center', va='center', size='x-small', c='w')
        text.set_bbox(dict(facecolor='black', alpha=0.5,pad=0.1))
        
        p = contour2piece(cs, img)
        _, _, cons = p
        plt.scatter(cons[0][:, 0], cons[0][:, 1], c='r', s=1)
        plt.scatter(cons[1][:, 0], cons[1][:, 1], c='g', s=1)
        plt.scatter(cons[2][:, 0], cons[2][:, 1], c='b', s=1)
        plt.scatter(cons[3][:, 0], cons[3][:, 1], c='k', s=1)
        
        pieces.append(p)
        count += 1
    
    
    print(f' > Processed file {path}. Pieces: ', count-count_offset)
    plt.savefig(os.path.join('images', f'{img_name}_{count_offset}-{count-1}.png'), dpi=300, format='png')
    #plt.show()

    
    return pieces



# display y axis flipped for convenience (puzzle right side up)
def printGrid(grid, x, y):
    y = (y_ext-1-y) % y_ext
    for x0, row in enumerate(grid):
        for y0, item in enumerate(np.flip(row)):
            if x==x0 and y==y0:
                print('X ', end='')
                continue
            if item: print('O ',end='')
            else: print('. ', end='')
        print()


# find matches given arbitrary sides (at least one)
def findMatches(right, bottom, left, top, database):
    match_scores = []
    
    for cand_ind, cand in enumerate(database):
        piece, _, _ = cand
        
        # check every orientation
        for i in range(4):
        
            score = 0
            count = 0
            if right is not None:
                right_side, right_male = right
                side, male = piece[i]
                if right_male == male: continue
                score += splineDist(right_side, side)
                count += 1
                
            if bottom is not None:
                bottom_side, bottom_male = bottom
                side, male = piece[(i+1)%4]
                if bottom_male == male: continue
                score += splineDist(bottom_side, side)
                count += 1
                
            if left is not None:
                left_side, left_male = left
                side, male = piece[(i+2)%4]
                if left_male == male: continue
                score += splineDist(left_side, side)
                count += 1
                
            if top is not None:
                top_side, top_male = top
                side, male = piece[(i+3)%4]
                if top_male == male: continue
                score += splineDist(top_side, side)
                count += 1
                
            score /= count
            match_scores.append([score, cand_ind, i])
            
    return match_scores



## continuous puzzle solver

x_ext = 28
y_ext = 36
grid_solved = np.zeros((x_ext, y_ext), dtype=bool)

"""
# provide solved positions that are not at the frontier
grid_solved[0, 0:6] = True
grid_solved[1, 0:4] = True
grid_solved[2, 0:2] = True
grid_solved[3:7, 0] = True
"""



use_cache = True

# read database images
db_file = 'database.pickle'
if os.path.exists(db_file) and use_cache:
    with open(db_file, 'rb') as handle:
        database = pickle.load(handle)
        print(' > database loaded from cache')
else:
    print('creating database...')
    database_path = os.path.join('images/database/')
    database_files = []
    for image in os.scandir(database_path):
        _, file = os.path.split(image.path)
        _, ext = os.path.splitext(file)
        if ext == '.jpg':
            database_files.append(image.path)

    database = []
    for path in database_files:
        database.extend(img2pieces(path, len(database), 'database'))

    with open(db_file, 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('database saved.')
        

# read setup files (already matched puzzle pieces)
setup_pieces = []
setup_file = 'setup.pickle'
if os.path.exists(setup_file) and use_cache:
    with open(setup_file, 'rb') as handle:
        setup_pieces = pickle.load(handle)
        print(' > setup loaded from cache')
else:
    setup_path = os.path.join('images/setup')
    # paths to img of pieces for grid setup
    setup_imgs = ['top_left_corner.jpg']
    for file_name in setup_imgs:
        path = os.path.join(setup_path, file_name)
        setup_pieces.extend(img2pieces(path, len(setup_pieces), 'setup'))
    
    with open(setup_file, 'wb') as handle:
        pickle.dump(setup_pieces, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('setup saved.')


# for each img above, provide list of (x, y, offset) position of the puzzle pieces on the img, and orientation offset.
# x is top-down, y is left-right (array notation)
# offset: s.t. puzzle side order from image: right, bottom, left, top
setup_config = {   
    # start top_left_corner.jpg
    0: (8, 0, 0),
    1: (7, 0, 0),
    2: (6, 0, 0),
    3: (6, 1, 0),
    4: (5, 1, 0),
    5: (5, 0, 0),
    6: (4, 1, 0),
    7: (4, 0, 0),
    8: (3, 1, 0),
    9: (3, 0, 0),
    10: (2, 1, 0),
    11: (2, 0, 0),
    12: (2, 2, 0),
    13: (2, 3, 0),
    14: (1, 1, 0),
    15: (1, 5, 0),
    16: (1, 2, 0),
    17: (1, 3, 0),
    18: (1, 4, 0),
    19: (1, 0, 0),
    20: (0, 1, 0),
    21: (0, 2, 0),
    22: (0, 0, 0),
    23: (0, 3, 0),
    24: (0, 5, 1),
    25: (0, 4, 0),
    # end top_left_corner.jpg
}



grid_pieces = {}


# load puzzle
use_puzzle_cache = True # Dont't change or looses progress!!!!!
puzzle_file = 'puzzle.pickle'
if os.path.exists(puzzle_file) and use_puzzle_cache:
    with open(puzzle_file, 'rb') as handle:
        grid_pieces, grid_solved = pickle.load(handle)
        print(' > puzzle loaded from cache')


def savePuzzle(grid_pieces, grid_solved):
    with open(puzzle_file, 'wb') as handle:
        pickle.dump((grid_pieces, grid_solved), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('puzzle saved.')

# roll list with wrap around
def roll(l, offset):
    return l[-offset:] + l[:-offset]
    

# add new setup pieces to grid
for i in range(len(setup_pieces)):
    x, y, offset = setup_config[i]
    y = (y_ext-1-y) % y_ext # here this compensates for having flipped y coordinates in setup data (puzzle view to internal view)
    if (x, y) in grid_pieces: continue # skip existing entries
    grid_solved[x, y] = True
    piece, _, _ = setup_pieces[i]
    grid_pieces[(x, y)] = roll(piece, offset)





# return (number of neighbours, right, bottom, left, top)
def getNeighbours(grid, grid_pieces, x, y):
    max_x = grid.shape[0]
    max_y = grid.shape[1]
    
    num = 0
    right = None
    bottom = None
    left = None
    top = None
    
    # top
    if x - 1 >= 0 and grid[x-1, y]:
        top = grid_pieces[(x-1, y)][1]
        num += 1
    # right
    if y + 1 < max_y and grid[x, y+1]:
        right = grid_pieces[(x, y+1)][2]
        num += 1
    # bottom
    if x + 1 < max_x and grid[x+1, y]:
        bottom = grid_pieces[(x+1, y)][3]
        num += 1
    # left
    if y - 1 >= 0 and grid[x, y-1]:
        left = grid_pieces[(x, y-1)][0]
        num += 1
    
    return (num, right, bottom, left, top)
    

# iterate over grid and find next match
for x in range(grid_solved.shape[0]):
    for y in range(grid_solved.shape[1]):
    
        #if x != 4 or y != (y_ext-1-1) % y_ext: continue
        if grid_solved[x, y]: continue
        num, right, bottom, left, top = getNeighbours(grid_solved, grid_pieces, x, y)
        
                
        if num >= 2:
            printGrid(grid_solved, x, y)
            print(f'> finding best {num}-side matches at {x}, {(y_ext-1-y) % y_ext}:')
            
            match_scores = findMatches(right, bottom, left, top, database)
            
            
            match_scores = np.asarray(match_scores)
            match_scores = match_scores[np.argsort(match_scores[:, 0])]
            
            print ("{:<10} {:<10} {:<10}".format('Score','Piece','Orientation'))
            for match in match_scores[:5]:
                score, piece, offset = match
                offset = int(offset)
                print ("{:<10} {:<10} {:<10}".format(round(score, 3), int(piece), offset))
            
                
            
            # wait for user to confirm valid piece
            print('input piece number to save (-1: skip, -2: exit): ')
            selected_piece_ind = str(input())
            
            while selected_piece_ind == "s":
                print('which entry to show?')
                nr = int(input())
                if nr < 0: break
                
                best_score, best_piece_ind, best_side_ind = match_scores[nr]
                best_piece_ind = int(best_piece_ind)
                best_side_ind = int(best_side_ind)
                best_cand = database[best_piece_ind]
                piece, sides, contours = best_cand
                
                
                
                f, axs = plt.subplots(2, 2)
                
                
                axs[0, 0].set_aspect('equal')
                axs[0, 1].set_aspect('equal')
                axs[1, 0].set_aspect('equal')
                axs[1, 1].set_aspect('equal')
                    
                
                # queries
                if right is not None:
                    axs[0, 0].scatter(right[0][:, 0], right[0][:, 1], c='b', label='query right')
                if bottom is not None:
                    axs[0, 1].scatter(bottom[0][:, 0], bottom[0][:, 1], c='b', label='query bottom')
                if left is not None:
                    axs[1, 0].scatter(left[0][:, 0], left[0][:, 1], c='b', label='query left')
                if top is not None:
                    axs[1, 1].scatter(top[0][:, 0], top[0][:, 1], c='b', label='query top')
                    
                
                # match
                axs[0, 0].scatter(piece[best_side_ind][0][:, 0], piece[best_side_ind][0][:, 1], c='g', label='match right')
                axs[0, 1].scatter(piece[(best_side_ind+1)%4][0][:, 0], piece[(best_side_ind+1)%4][0][:, 1], c='g', label='match bottom')
                axs[1, 0].scatter(piece[(best_side_ind+2)%4][0][:, 0], piece[(best_side_ind+2)%4][0][:, 1], c='g', label='match left')
                axs[1, 1].scatter(piece[(best_side_ind+3)%4][0][:, 0], piece[(best_side_ind+3)%4][0][:, 1], c='g', label='match top')
                
                
                axs[0, 0].legend()
                axs[0, 1].legend()
                axs[1, 0].legend()
                axs[1, 1].legend()
                plt.show()
                
                
                
            print('input piece number to save (-1: skip, -2: exit): ')
            selected_piece_ind = str(input())
                
            selected_piece_ind = int(selected_piece_ind)
            if selected_piece_ind == -1:
                print('skipping this piece...')
                continue
            if selected_piece_ind < 0:
                print('exit.')
                exit()
            
            def side2genderStr(side):
                if side is not None:
                    return str(side[1])
                else:
                    return '_'
            
            
            # save piece
            grid_solved[x, y] = True
            selected_piece, _, _ = database[selected_piece_ind]
            selected_piece = roll(selected_piece, offset)
            print('neighbour sides: ', side2genderStr(right), side2genderStr(bottom), side2genderStr(left), side2genderStr(top))
            print('piece sides: ', side2genderStr(selected_piece[0]), side2genderStr(selected_piece[1]), side2genderStr(selected_piece[2]), side2genderStr(selected_piece[3]))
            grid_pieces[(x, y)] = selected_piece
            savePuzzle(grid_pieces, grid_solved)
            

















