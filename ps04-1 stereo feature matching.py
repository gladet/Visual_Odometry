    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'''
 SANTA CLARA UNIVERSITY COMPUTER VISION I 
 PROGRAMMING ASSIGNMENT 4-1
 MARCH 2018
 
 Joe Weber
 '''
"""

from skimage.feature import (match_descriptors, corner_harris, corner_shi_tomasi,
                             corner_peaks, corner_kitchen_rosenfeld, ORB, BRIEF, plot_matches)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.color import (rgb2gray, gray2rgb)


''' HAMMING DISTANCE
returns the hamming distance between two equal sized binary vectors
'''
def hamming_distance(p1,p2):
    return np.count_nonzero(p1!=p2) #so 1 for != and 0 for == -> hamming dist is to count the degree of difference

'''     DISTANCE MATCH
example function that matches keypoints in a left and right image by finding the closest
match by distance in image coordinates.
'''
def dist_match(l_keypoints, r_keypoints, l_descript, r_descript, cam, y_thresh): 
    # two sets of feature points -> what is the descriptor? 
    
    matches = []    
    # loop over every match in the left list of keypoints
    for lidx,lk in enumerate(l_keypoints): #left index / left keypoint -> enum used in prev assignment
        best_score = 100000 #like int_max
        best_idx = -1
        # now look at every keypoint in right list and find best score (distance)
        # note that I use a loop to iterate over the points. The more 'pythonic' way
        # doesnt require a loop
        for ridx,rk in enumerate(r_keypoints):
            deltax = rk[1]-lk[1] # note that keypoints are (y,x) pairs
            deltay = rk[0]-lk[0] # y is row / x is col
            distance = np.sqrt(deltax**2 + deltay**2) 
            if distance < best_score:
                # found a better match, so save the score and index
                best_idx = ridx
                best_score = distance # update best_score / best_idx
        # if found a match, save the matching index numbers.
        if best_idx > -1:
            matches.append([lidx,best_idx]) #append just like arraylist.add in java

    return matches

'''     EPIPOLAR MATCH
matches keypoints in a left and right image under both the epipolar constraint
and the assumption of a planar surface which provides a lower bound on disparity as
function of y coordinate. Then finds all possible matches on that line and
finds the one with the lowest hamming distance (best match)
TODO: Complete this
'''

def epipolar_match(l_keypoints, r_keypoints, l_descript, r_descript, cam, y_thresh):
    fx = cam['fx']
    fy = cam['fy']
    baseline = cam['baseline']
    camera_height = cam['camera_height']
    max_disparity = 80
    
    matches = []
    for lidx,lk in enumerate(l_keypoints):
        best_score = 256  # the largest hamming distance is 255
        best_idx = -1
        for ridx,rk in enumerate(r_keypoints):
            #TODO Fill in this area
            #    something = max(fx*baseline*(y-cy)/(fy*camera_height),0)
            #
            #
            matches.append([lidx,best_idx])
    return matches

# MAIN CODE
# TODO: set the paths to folders for KITTI Dataset:
root_pathname = "/Users/josephweber/Documents/Computer Vision/Image Data Sets/KITTIVO/"
image_folder = "images/"
calib_folder = "calibs/"
poses_folder = "poses/"
sequence_number = "00/"
left_camera = "cam00/"
right_camera = "cam01/"

# read calibration file
calib_path = root_pathname+calib_folder+sequence_number
calib_file = "calib.txt"

# Load the camera Calibration information

try:
    with open(calib_path+calib_file, 'r') as f:
       line = f.readline()
       K_left = np.fromstring(line[4:], dtype=float, sep=' ')
       K_left = K_left.reshape(3,4)
       line = f.readline()
       K_right = np.fromstring(line[4:], dtype=float, sep=' ')
       K_right = K_right.reshape(3,4)

except FileNotFoundError:
            print('Unable to open calibration file: ' + calib_file)
f.close()

baseline = -K_right[0,3]
fx = K_left[0,0]
fy = K_left[1,1]
cx = K_left[0,2]
cy = K_left[1,2]
camera_height = 1600 # 1.6 meters

# create a dict of camera information
camera_data = dict([("fx", fx)])
camera_data["fy"]=fy
camera_data["cx"]=cx
camera_data["cy"]=cy
camera_data["baseline"]=baseline
camera_data["camera_height"]=camera_height

frame = 0


vertical_thresh = 5 # max number of pixels up or down allowed in epipolar matching (ideal epipolar has zero pixels up/down)

frame_number = '000'+str(frame//100)+str(frame//10)+str(frame%10)

# load the image into a NUMPY array using matplotlib's imread function
left_img_file = root_pathname+image_folder+sequence_number+left_camera+frame_number+'.png'
l_image = plt.imread(left_img_file)
right_img_file = root_pathname+image_folder+sequence_number+right_camera+frame_number+'.png'
r_image = plt.imread(right_img_file)

# find Harris corner features in each camera
l_keypoints = corner_peaks(corner_harris(l_image), min_distance=10)
r_keypoints = corner_peaks(corner_harris(r_image), min_distance=10)
# TODO Replace the two lines above with the Shi-Tomasi detector

# for each corner found, extract the BRIEF descriptor
extractor = BRIEF(sigma=1.0)
extractor.extract(l_image, l_keypoints)
l_descriptors = extractor.descriptors

# not all keypoints get descriptors. Remove the ones that didn't:
mask = extractor.mask
l_keypoints = l_keypoints[mask]
   
extractor.extract(r_image, r_keypoints)
r_descriptors = extractor.descriptors
mask = extractor.mask
r_keypoints = r_keypoints[mask]
  
# plot the found keypoints on top of the left image
fig, ax = plt.subplots(figsize=(20,5))
plt.imshow(l_image, cmap=cm.gray)
plt.plot(l_keypoints[:,1],l_keypoints[:,0],'+',r_keypoints[:,1],r_keypoints[:,0],'+')
plt.show()

#save the image
filename = "keypoints.png"
fig.savefig(filename)   # save the figure to file
plt.close(fig)

# do brute force matching between left and right
matcheslr = match_descriptors(l_descriptors, r_descriptors, cross_check=True)
print("Number of brute force l-r matches in frame ",frame," :",matcheslr.shape[0])
fig, ax = plt.subplots(figsize=(20,5))
plot_matches(ax, l_image, r_image, l_keypoints, r_keypoints, matcheslr,keypoints_color=[1,0,1])
plt.show()

filename = "l-r-brute-matches.png"   
fig.savefig(filename)   # save the figure to file
plt.close(fig)

# Do epipolar constrained matching
matcheslr = epipolar_match(l_keypoints, r_keypoints, l_descriptors, r_descriptors, camera_data, vertical_thresh)
matcheslr = np.array(matcheslr)    
fig, ax = plt.subplots(figsize=(20,5))
plot_matches(ax, l_image, r_image, l_keypoints, r_keypoints, matcheslr,keypoints_color=[1,0,1])
plt.show()

filename = "l-r-epipolar-matches.png"   
fig.savefig(filename)   # save the figure to file
plt.close(fig)
