import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import math
import pickle
from sklearn.decomposition import PCA

import scipy as sc


def side2genderStr(side):
    if side is not None:
        if side[1]: return 'M'
        return 'F'
    return '_'

# rotates 2D points (2 x n) about theta around the origin
def rotatePoints(points, theta):
    R = np.asarray([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return np.matmul(R, points) # rotate

# normalize puzzle side, such that reference points p1, p2 lie on (-1, 0) an (1, 0) respectively
# female contours gets rotated 180° to be comparable to males
def normalizeSide(contour):
    p1 = contour[0]
    p2 = contour[-1]


    side = contour
    side = side - (p1 + (p2-p1)/2) # center
    
    theta = -np.arctan2((p2[1]-p1[1]), (p2[0]-p1[0])) # rotation angle
    side = rotatePoints(side.T, theta).T
    
    denom = (abs(np.amax(side[:, 0]) - np.amin(side[:, 0])))*2
    
    if denom == 0:
        print('division by 0 error')
    side = side / (abs(np.amax(side[:, 0]) - np.amin(side[:, 0])))*2 # stretch to [-1, 1] on x-axis
    
    #plt.clf()
    #plt.scatter(side[:,0], side[:,1])
    #plt.show()
    
    
    male = False
    # check if male/female side        
    if np.amin(side[:, 1]) < -0.25:
        # female -> rotate 180°
        male = True
        side = rotatePoints(side.T, np.pi).T
    
    return side, male



# given sequence of 2d points and number of interpolation steps, return equidistantly interpolated spline
# returning "steps"-number of points (including original end-points), requires steps >= 2
def interpolateSpline(spline, steps):
    
    if steps < 2: raise Exception('Need at least two substeps for spline interpolation.')
    l = len(spline)
    if l < 2: raise Exception('Spline must have at least 2 points')
        
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
    l = len(s0)
    
    acc = 0
    for p0 in s0:
        min_dist = float('inf')
        for p1 in s1:
            dist = np.linalg.norm(p0-p1)
            if dist < min_dist:
                min_dist = dist
                
        acc += min_dist
    return acc / l
    

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
    x = s*(s-s1)*(s-s2)*(s-s3)
    if x < 0: return 0
    return np.sqrt(x)
    
    
def updateSideDivision(cs, inds):
    inds = sorted(inds)
    contours = []
    contours.append(cs[inds[0]:inds[1]])
    contours.append(cs[inds[1]:inds[2]])
    contours.append(cs[inds[2]:inds[3]])
    contours.append(cs.take(range(inds[3],len(cs) + inds[0]), mode='wrap', axis=0))
    return contours

# returns score of how good the corners represent the puzzle piece.
# area spanned and right angle measure of corners
# assumes points are ordered counterclockwise
# higher is better, 0 is worst
def corner_score(c1, c2, c3, c4, img):
    # side lengths
    s12 = np.linalg.norm(c1-c2)
    s23 = np.linalg.norm(c2-c3)
    s31 = np.linalg.norm(c3-c1)
    s34 = np.linalg.norm(c3-c4)
    s41 = np.linalg.norm(c4-c1)
    
    
    # if 2 points collide, abort
    if not s12 or not s23 or not s34 or not s41:
        return 0
    
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
    
    return area + anglescore + sideRatioScore

# args: all contour points, side 0, side 1
# returns index of best corner wrt cs
def refineCorner(cs, side0, side1):

    pca = PCA(n_components=2)
    start_f = 0.025
    end_f = 0.1
    
    # first corner 0-1
    l1 = len(side1)
    start = int(start_f * l1)
    end = int(end_f * l1)
    pca.fit(side1[start:end])
    vec1 = 10*pca.components_[0]
    mean1 = np.mean(side1[start:end], axis=0)
    plt.plot([mean1[0], mean1[0] + vec1[0]], [mean1[1], mean1[1] + vec1[1]], c='w')
    
    
    l0 = len(side0)
    start = int((1-end_f) * l0)
    end = int((1-start_f) * l0)
    pca.fit(side0[start:end])
    vec0 = 10*pca.components_[0]
    mean0 = np.mean(side0[start:end], axis=0)
    plt.plot([mean0[0], mean0[0] + vec0[0]], [mean0[1], mean0[1] + vec0[1]], c='r')

    
    i_x, i_y = lineLineIntersection(mean1, mean1+vec1, mean0, mean0+vec0)
    
    
    plt.scatter(i_x, i_y, c='g')
    ideal_p = np.asarray([i_x, i_y])
    
    # find closest point on contour
    min_dist = float('inf')
    refined_cand_ind = 0
    for i, p in enumerate(cs):
        dist = np.linalg.norm(p-ideal_p)
        if dist < min_dist:
            min_dist = dist
            refined_cand_ind = i
            
    return refined_cand_ind


# adapted from (copyright) https://gist.github.com/shubhamwagh/b8148e65a8850a974efd37107ce3f2ec
def smoothContour(contour):
    #plt.clf()
    #plt.scatter(contour[:, 0], contour[:, 1], c='b', s=20)
    
    x,y = contour[:, 0], contour[:, 1]
    tck, u = sc.interpolate.splprep([x,y], u=None, s=len(contour)//2, per=1)
    u_new = np.linspace(u.min(), u.max(), len(contour))
    x_new, y_new = sc.interpolate.splev(u_new, tck, der=0)
    res_array = [[int(i), int(j)] for i, j in zip(x_new, y_new)]
    contour = np.asarray(res_array, dtype=np.int32)

    return contour
    #plt.scatter(contour[:, 0], contour[:, 1], c='g', s=10)
    #plt.show()
    #exit()

def contour2piece(contour, img):
    
    # smooth initial contour
    cs = smoothContour(contour)

    # find corner candidates
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

    cornerCandidates = cs[maxima]

    
    # prune corners to best 4
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

    best_corners_inds = maxima[best_corners_inds]
    plt.scatter(best_corners[:, 0], best_corners[:, 1], s=10, facecolors='none', edgecolors='g')


    contours = updateSideDivision(cs, best_corners_inds)




    # refine corners    
    refined_corners_inds = [refineCorner(cs, contours[(i+3)%4], contours[i]) for i in range(4)]
    contours = updateSideDivision(cs, refined_corners_inds)
    refined_corners = cs[refined_corners_inds]
    
    
    # show corners
    #plt.scatter(cornerCandidates[:, 0], cornerCandidates[:, 1], c='r')
    plt.scatter(best_corners[:, 0], best_corners[:, 1], facecolors='none',edgecolors='b')
    plt.scatter(refined_corners[:, 0], refined_corners[:, 1], c='y', s=100)
    
    # display contours
    plt.scatter(contours[0][:, 0], contours[0][:, 1], c='r', s=1)
    plt.scatter(contours[1][:, 0], contours[1][:, 1], c='g', s=1)
    plt.scatter(contours[2][:, 0], contours[2][:, 1], c='b', s=1)
    plt.scatter(contours[3][:, 0], contours[3][:, 1], c='k', s=1)
   
    #plt.show()
    #exit()
    
    sides = [normalizeSide(cont) for cont in contours]
    
    piece = []
    for side, male in sides:
        spline = interpolateSpline(side, 100)
        # ignore boundaries as the corners are not sharp
        truncated, _ = normalizeSide(spline[3:-3])
        
        piece.append((truncated, male))
    
    
    return piece, sides, contours


def img2pieces(path, count_offset, img_name):
    img = cv2.imread(path)
    img = cv2.medianBlur(img, ksize=5)

    # prep image
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gray   = cv2.medianBlur(gray, ksize=5)
    #edged = cv2.Canny(blurred, 100, 255)
    
    #blurred_edge = cv2.GaussianBlur(edged, (3, 3), 0)
    
    # define a (3, 3) structuring element
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    #kernel = np.ones((5,5),np.uint8)
    #closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    #dilate = cv2.dilate(gradient, kernel, iterations=1)
    #erosion = cv2.erode(dilate,kernel,iterations = 2)
    
    # apply the dilation operation to the edged image
    
    
    
    
    #thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.blur(thresh, ksize=(3, 3))
    
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,5)
    th3 = cv2.blur(th3, ksize=(3, 3))
    
    
    f, axs = plt.subplots(2, 2)
    axs[0,0].imshow(img[:, :, 2], cmap='gray')
    axs[0,1].imshow(th3)
    axs[1,0].hist(img[:, :, 2].flatten(), bins=100, log=False)
    plt.show()
    exit()
    
    
    plt.imshow(img)
    
    contours, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours, _ = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
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
        
        # not to far off a square
        if w > 2*h or h > 2*w: continue
        
        area = cv2.contourArea(contours[i])
        if area < 20000: continue
        #if area > 100000: continue
        
        """
        if count != 16:
            count += 1
            continue
        """
        cs = np.squeeze(np.asarray(contours[i]), axis=1)
        
        center_x, center_y = np.mean(cs, axis=0)
        text = plt.text(center_x, center_y, f'{count}', ha='center', va='center', size='x-small', c='w')
        text.set_bbox(dict(facecolor='black', alpha=0.5,pad=0.1))
        
        p = contour2piece(cs, img)
        
        pieces.append(p)
        count += 1
    
    
    print(f' > Processed file {path}. Pieces: ', count-count_offset)
    #plt.savefig(os.path.join('images', f'{img_name}_{count_offset}-{count-1}.png'), dpi=300, format='png')
    plt.show()
    exit()

    
    return pieces
    
    
######################### start of main part #####################################################################


path = os.path.join('images/reconstruction/')
files = []
for image in os.scandir(path):
    _, file = os.path.split(image.path)
    _, ext = os.path.splitext(file)
    if ext == '.jpg':
        files.append(image.path)

database = []
for path in files:
    database.extend(img2pieces(path, len(database), 'database'))
    
    
    
    
    
    
    
    
    
    
