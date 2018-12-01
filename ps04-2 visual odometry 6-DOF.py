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

from skimage.feature import (match_descriptors, corner_harris,corner_shi_tomasi,
                             corner_peaks, ORB, BRIEF, plot_matches, corner_fast)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.color import (rgb2gray, gray2rgb)
import os
import scipy.optimize as sciopt
import math

# Global Variables -> not defined in functions

show_images = False #True #True #False #True #True
save_images = False
speed_thresh = 40 # max number of pixels a point can move between time frames due to rotaton or translation alone
vertical_thresh = 4 #8 #2 #4 # max number of pixels up or down allowed in epipolar matching (ideal epipolar has zero pixels up/down)
score_thresh = 64 #96 #32 #64 # maximum hamming distance between points to be considered a match
outlier_threshold = 100 # threshold on error residuals for determining outliers
min_peak_dist = 10 # minimum distance between detected corner features

frame_start = 100 #300 #500 #700 #0 #80 #4 #100 #80 #100 #4 #80 #0 #80 #100 #4 #80 #100 #4 #0
frame_end = 300 #500 #700 #900 #999 #120 #6 #102 #120 #102 #6 #120 #999 #120 #102 #6 #120 #102 #6 #10

''' INVERSE_PROJECTION
returns the world 3D coordinates from an image point and its disparity
'''
def inverse_projection(img_pt, disparity, cam):
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    cy = cam['cy']
    baseline = cam['baseline']

    Z = fx*baseline/disparity
    X = (l_img_point[1]-cx)*Z/fx
    Y = -(l_img_point[0]-cy)*Z/fy
    return X,Y,Z


''' HAMMING DISTANCE
returns the hamming distance between two equal sized binary vectors
'''
def hamming_distance(p1,p2):
    return np.count_nonzero(p1!=p2)

'''     EPIPOLAR MATCH
matches keypoints in a left and right image under both the epipolar constraint
and the assumption of a planar surface which provides a lower bound on disparity as
function of y coordinate
'''
def epipolar_match(l_keypoints, r_keypoints, l_descript, r_descript, cam, y_thresh):
    fx = cam['fx']
    fy = cam['fy']
    baseline = cam['baseline']
    camera_height = cam['camera_height']
    
    matches = []
    dmax = 1.5*fx*baseline*cy/(fy*camera_height)
    for lidx,lk in enumerate(l_keypoints):
        best_score = score_thresh # must be less than this to be considered a match
        best_idx = -1
        for ridx,rk in enumerate(r_keypoints):
            deltay = abs(rk[0]-lk[0])
            if deltay < y_thresh:
                disparity = lk[1]-rk[1]
                y = rk[0]
                d_thresh = max(dmax*(y-cx),0)
                if disparity >d_thresh and disparity < d_thresh+dmax:
                    score = hamming_distance(l_descript[lidx],r_descript[ridx])
                    if score < best_score:
                        best_idx = ridx
                        best_score = score
        if best_idx>-1:
            matches.append([lidx,best_idx])
    return matches
#'''
 
'''     FORWARD MOTION MATCH
matches keypoints in time separated images under the assumpption of forward motion and
rotation about the vertical axis
'''                   
def forward_motion_match(l_keypoints, r_keypoints, l_descript, r_descript, cam,speed_thresh): #not to match l/r but prev/curr frames
    cx = cam['cx']
    cy = cam['cy']
    center = np.array([cy, cx])
    
    matches = []
    for lidx,lk in enumerate(l_keypoints):
        best_score = score_thresh # must be less than this to be considered a match
        best_idx = -1
        d_to_center = np.sqrt( np.sum ( (lk-center)*(lk - center) ) ) #dist to center
   
        for ridx,rk in enumerate(r_keypoints):
            distance = np.sqrt( np.sum( (rk-lk)*(rk-lk) ) ) #check L-2 dist
    
            if distance < speed_thresh*(1+d_to_center/fx): #what is the meaning of (1+d_to_center/fx)?
                score = hamming_distance(l_descript[lidx],r_descript[ridx])
                if score < best_score:
                    best_idx = ridx
                    best_score = score
                
        if best_idx>-1:
            matches.append([lidx,best_idx]) #find left / right match
    return matches
        
'''     DETERMINE MOTION 
Given 2D points in the image from previous time, and corresponding 3D points. Find the motion 
that makes the re-projection of the 3D points match as closely as possibl the 2D points
'''
def determine_motion(image_x, image_y, pts_3D, cam):
        # project the 3D points back into new left image and compare to found matches in new left image
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    cy = cam['cy']
    
    # Apply Motion
    # Internal function that applies 3D rotation and translation to pts_3D array
    def apply_motion(motion):
        #theta, d = motion
        #theta, tx, tz = motion
        rotx, roty, rotz, tx, ty, tz = motion
        # rotate and translate the 3D points
        cx, sx = np.cos(rotx), np.sin(rotx)
        cy, sy = np.cos(roty), np.sin(roty)
        cz, sz = np.cos(rotz), np.sin(rotz)
        RT = np.array([[ cy * cz, -cy * sz, sy, tx],
                       [ sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy, ty], #ty
                       [ -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy, tz],
                       [ 0, 0, 0,  1]]) #[R/T]
        # move the 3D points. First make points homogenouse by adding ones
        ho = np.insert(pts_3D, 3, 1, axis=0)
        new_pts_3D = RT.dot(ho) #new pts after transformation
        # convert back to cartesian
        new_pts_3D = np.delete(new_pts_3D, 3, 0)
        return new_pts_3D
    
    # Project_3D_to_2D
    # Projects 3D points to 2D image points on left camera
    def project_3D_to_2D(p3D):    
        r_xs = fx*p3D[0,:]/p3D[2,:] + cx # X / Z
        r_ys = -fy*p3D[1,:]/p3D[2,:] + cy # Y / Z
        return r_xs, r_ys
    
    # err_function
    # Calculates the reprojection error afer motion is applied to points
    def err_function(motion):
        new_pts_3D = apply_motion(motion)

        # calc new projections into image
        r_xs, r_ys = project_3D_to_2D(new_pts_3D)
        # calc re-projection error
        residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
        ssderror = np.sqrt( np.sum( residuals ) ) # sum squared diff -> then sqrt -> square root
        return ssderror
    
    # start of function
    if show_images: # show image
        # draw the two sets of points
        r_xs, r_ys = project_3D_to_2D(pts_3D) #the prev frame key points -> 3D -> reproj
        fig, ax = plt.subplots(figsize=(20,5))
        ax.set_title('Optical Flow')
        plt.plot(r_xs,r_ys,'+',image_x,image_y,'+') #show image pixels
        for i in range(len(r_xs)):
            plt.plot([image_x[i], r_xs[i]],[image_y[i],r_ys[i]],'-') #show line between two pels
        plt.show()
        plt.close(fig)
        residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
        print("initial error: ",np.sqrt( np.sum( residuals )))
   
    # initial estimate is no motion:
    #motion0 = [0,0]  # equals rotation angle theta, and translation d
    #motion0 = [0,0,0]  # equals rotation angle theta, and translation tx / tzs
    motion0 = [0,0,0,0,0,0]  # equals rotation angle theta, and translation tx / tzs
    
    # set bounds on possible rotation and motion between frames (no reversing allowed!)
    #bounds = ( (-math.pi/18, math.pi/18 ), (-2000,0))
    #bounds = ( (-math.pi/18, math.pi/18 ), (-2000,0), (-2000,0))
    #bounds = ( (-math.pi/18, math.pi/18 ), (-math.pi/18, math.pi/18 ), (-math.pi/18, math.pi/18 ), (-2000,0), (-2000,0), (-2000,0))
    #bounds = ( (-math.pi/90, math.pi/90 ), (-math.pi/18, math.pi/18 ), (-math.pi/90, math.pi/90 ), (-1500,0), (-100,0), (-1500,0))
    bounds = ( (-math.pi/45, math.pi/45 ), (-math.pi/18, math.pi/18 ), (-math.pi/45, math.pi/45 ), (-1500,0), (-100,0), (-1500,0))
        
    res = sciopt.minimize(err_function, motion0, method = "SLSQP", bounds = bounds) #do optimization
    motion = res.x #what is motion here?
    print(res.message)
    print("Minimization error: ", err_function(motion))
    
    # calculate residuals
    new_pts_3D = apply_motion(motion) 
    r_xs, r_ys = project_3D_to_2D(new_pts_3D)
    residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
    print("residuals: ", np.sqrt( np.sum( residuals )))
    print("Minimization error: ", err_function(motion))
    if show_images:
        plt.plot(residuals,'o')
        plt.show()
 
    #remove outliers and re-calculate motion
    mask = residuals<outlier_threshold
    print("Keeping ", np.sum(mask), " non-outliers out of ", len(image_x))
    image_x = image_x[mask]
    image_y = image_y[mask]
    pts_3D = pts_3D[:,mask]
    motion0 = motion
    res = sciopt.minimize(err_function, motion0, method = "SLSQP", bounds = bounds)
    motion = res.x
    print(res.message)
    print("Re-Minimization error: ", err_function(motion))
    #'''
    # calculate residuals again
    new_pts_3D = apply_motion(motion) 
    r_xs, r_ys = project_3D_to_2D(new_pts_3D)
    residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
    if show_images:
        plt.plot(residuals,'o')
        plt.show()
   
    #remove any remaining outliers a SECOND TIME and re-calculate motion if any
    mask = residuals<outlier_threshold/2 #1/2 of prev thresh
    print("Keeping ", np.sum(mask), " non-outliers out of ", len(image_x))
    if len(image_x) - np.sum(mask) > 0: #num of outliers > 0
        #TODO: fill in this section
        
        image_x = image_x[mask]
        image_y = image_y[mask]
        pts_3D = pts_3D[:,mask]
        motion0 = motion
        res = sciopt.minimize(err_function, motion0, method = "SLSQP", bounds = bounds)
        motion = res.x
        print(res.message)
        print("2nd Re-Minimization error: ", err_function(motion))
         
    #'''
    # draw the final two sets of points
    if show_images:
        new_pts_3D = apply_motion(motion) 
        r_xs, r_ys = project_3D_to_2D(new_pts_3D)
        fig, ax = plt.subplots(figsize=(20,5))
        ax.set_title('Reprojection Error')
        plt.plot(r_xs,r_ys,'+',image_x,image_y,'+')
        for i in range(len(r_xs)):
            plt.plot([image_x[i], r_xs[i]],[image_y[i],r_ys[i]],'-')
        plt.show()
        plt.close(fig)
        residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
        plt.plot(residuals,'o')
        plt.show()
        print("Final Re-Minimization error: ", err_function(motion))
        
    #return motion[0], motion[1]
    #return motion[0], motion[1], motion[2]
    return motion[0], motion[1], motion[2], motion[3], motion[4], motion[5]

'''        
# MAIN CODE
# TODO: set the paths to folders for KITTI Dataset:
'''
#root_pathname = "/Users/josephweber/Documents/Computer Vision/Image Data Sets/KITTIVO/"
root_pathname = "/Users/gladet/courses/coen344/hw4/KITTIVO/"    
image_folder = "images/"
calib_folder = "calibs/"
poses_folder = "poses/"
sequence_number = "00/"
left_camera = "cam00/"
right_camera = "cam01/"

# read camera calibration file
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

# extract camera intrinsics from K matrices
fx = K_left[0,0]
fy = K_left[1,1]
cx = K_left[0,2]
cy = K_left[1,2]
camera_height = 1600 # 1.6 meters
baseline = 540 # 54cm. It should equal -K_right[0,3] but doesn't

# create a dict of camera information
camera_data = dict([("fx", fx)])
camera_data["fy"]=fy
camera_data["cx"]=cx
camera_data["cy"]=cy
camera_data["baseline"]=baseline
camera_data["camera_height"]=camera_height

# Create an array for storing our motion estimates for every pair of frames
# consisting of delta heading (angle) and forward motion
#motion_estimates = np.zeros((2,frame_end-frame_start)) # 6 - 4 = 2 -> 2 motions -> R | T
#motion_estimates = np.zeros((3,frame_end-frame_start)) 
motion_estimates = np.zeros((6,frame_end-frame_start)) 

# placeholders:
previous_r_img = []
previous_r_keypoints = []
previous_r_desc = []
previous_l_img = []
previous_l_keypoints = []
previous_l_desc = []
previous_lr_matches = []

# loop over images
frame = frame_start # 4
while frame<=frame_end: # end: 6

    frame_number = str(frame).zfill(6) #000004 -> 000006
    print("Working on images ", frame_number," up to ", frame_end)
    
    # load the image into a NUMPY array using matplotlib's imread function
    left_img_file = root_pathname+image_folder+sequence_number+left_camera+frame_number+'.png'
    l_image = plt.imread(left_img_file)
    right_img_file = root_pathname+image_folder+sequence_number+right_camera+frame_number+'.png'
    r_image = plt.imread(right_img_file) #imread(filenameinstrformat)
    
    # find corner features in each camera
    l_keypoints = corner_peaks(corner_shi_tomasi(l_image), min_distance=min_peak_dist)
    r_keypoints = corner_peaks(corner_shi_tomasi(r_image), min_distance=min_peak_dist)
    
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
    if show_images:
        fig, ax = plt.subplots(figsize=(20,5))
        plt.imshow(l_image, cmap=cm.gray) #only show grayscale image?
        plt.plot(l_keypoints[:,1],l_keypoints[:,0],'+',r_keypoints[:,1],r_keypoints[:,0],'+')
        ax.set_title('Keypoints Detected')
        plt.show()
        
        #save the image
        if save_images:
            filename = "keypoints.png"
            fig.savefig(filename)   # save the figure to file
            plt.close(fig)
    
    # Do epipolar constrained matching
    matcheslr = epipolar_match(l_keypoints, r_keypoints, l_descriptors, r_descriptors, camera_data, vertical_thresh)
    matcheslr = np.array(matcheslr)  #find the matches
    
    if show_images:
        fig, ax = plt.subplots(figsize=(20,5))
        plot_matches(ax, l_image, r_image, l_keypoints, r_keypoints, matcheslr,keypoints_color='red', matches_color='yellow') #keypoints_color=[1,0,1])
        #plot_matches(ax, l_image, r_image, l_keypoints, r_keypoints, matcheslr,keypoints_color=[1,0,1])
        ax.set_title('Stereo Correspondences') #get the stereo correspondences
        plt.show()
        
        if save_images:
            filename = "l-r-epipolar-matches.png"   
            fig.savefig(filename)   # save the figure to file
            plt.close(fig)
    
    # now match features with previous frame
    if frame > frame_start: #start from the 2nd frame in the sequence
        matchesll = forward_motion_match(previous_l_keypoints, l_keypoints, previous_l_desc, l_descriptors, camera_data, speed_thresh)
        matchesll = np.array(matchesll)
        print("Number of l-l matches: ",matchesll.shape) #get the l-l matches for [R|T] estimation
        
        if show_images:
            fig, ax = plt.subplots(figsize=(20,5))
            plt.gray()
            plot_matches(ax, previous_l_img, l_image, previous_l_keypoints,  l_keypoints, matchesll,keypoints_color='red', matches_color='yellow') #keypoints_color=[1,0,1])
            ax.set_title('Temporal Correspondences')
            plt.show()
      
        # for every point in previous frame that we matched for stereo
        # see if we also matched to current frame in time 

        stereo_matches_in_previous = [m[0] for m in previous_lr_matches]
        # do they have matching in current time frame?
        time_matches = [m[0] for m in matchesll]
        overlap = list(set(stereo_matches_in_previous) & set(time_matches))
        print("Found ",len(overlap)," left stereo matches that mapped to new left frame")
        
        # now get 3D points from stereo match for those points with matches (in overlap)  
        min_disparity = 3
        num_p1 = 0
        time_matches = []
        
        # for every stereo match in last frame, look for overlap ones
        for m in previous_lr_matches:
            if m[0] in overlap:
                l_img_point = previous_l_keypoints[m[0]]
                r_img_point = previous_r_keypoints[m[1]]
                disparity = (l_img_point-r_img_point)[1]
                # for those with disparity> thresh find the associated time match l-l
                if disparity > min_disparity:
                    for mt in matchesll:
                        if mt[0] == m[0]:
                            time_matches.append(mt)
                    
                    num_p1 = num_p1+1
                    X,Y,Z = inverse_projection(l_img_point,disparity, camera_data)
                    if num_p1==1:
                        points_3D = np.array([[X],[Y],[Z]])
                    else:
                        points_3D = np.append(points_3D,[[X],[Y],[Z]],axis=1)
        
        # create lists of x and y locations in the current frame
        # for the points that have stereo matches and temporal matches
        xs=[]
        ys=[]
        for m in time_matches:
            idx = m[1];
            xs.append(l_keypoints[idx,1])
            ys.append(l_keypoints[idx,0])
        xs = np.array(xs)
        ys = np.array(ys)
      
        #rot, trans = determine_motion(xs, ys, points_3D, camera_data)
        #rot, tx, tz = determine_motion(xs, ys, points_3D, camera_data)
        rotx, roty, rotz, tx, ty, tz = determine_motion(xs, ys, points_3D, camera_data)
        trans = np.sqrt(tx ** 2 + tz ** 2)
        #print("Estimated rotation of :", -rot*180/math.pi, "degrees and translation of ",-trans/1000," meters.")
        print("Estimated rotation of :", -roty*180/math.pi, "degrees and translation of ",-trans/1000," meters.")
        #motion_estimates[:,frame-frame_start-1] = [-rot, -trans]
        #motion_estimates[:,frame-frame_start-1] = [-rot, -tx, -tz]
        motion_estimates[:,frame-frame_start-1] = [-rotx, -roty, -rotz, -tx, -ty, -tz]
            
    # end of if frame not first frame
    # copy current stuff to previous stuff
    previous_r_img = r_image
    previous_r_keypoints = r_keypoints
    previous_r_desc = r_descriptors
    
    previous_l_img = l_image
    previous_l_keypoints = l_keypoints
    previous_l_desc = l_descriptors
    
    previous_lr_matches = matcheslr
    
    frame = frame+1
# END Frame loop

        
# Get Poses for ground truth trajectory
pose_path = root_pathname+poses_folder
pose_file = os.path.join(pose_path, sequence_number[:-1] + '.txt')
poses = []
try:
    with open(pose_file, 'r') as f:
       lines = f.readlines()

    for line in lines:
        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
        T_w_cam0 = T_w_cam0.reshape(3, 4)
        poses.append( T_w_cam0 )

except FileNotFoundError:
            print('Unable to open pose file: ' + pose_file)

# extract the x and z coordinates from the ground truth poses 
x_pos = []
z_pos = []
# extract X and Z values
for i in range(frame_end-frame_start+1):
    x_pos.append(poses[i+frame_start][0,3])
    z_pos.append(poses[i+frame_start][2,3])

# estimate path of car and compare to ground truth
vehicle_state = np.zeros((3,frame_end-frame_start+1)) # heading (angle), x and z position
#vehicle_state = np.zeros((6,frame_end-frame_start+1)) # heading (angle), x and z position
# match our first position and direction to ground truth
# get initial heading from the frist two translations in x-z plane from ground truth
theta = math.atan2(x_pos[1]-x_pos[0], z_pos[1]-z_pos[0])
#thetay = math.atan2(x_pos[1]-x_pos[0], z_pos[1]-z_pos[0])
vehicle_state[:,0] = [theta, 1000*x_pos[0], 1000*z_pos[0]]
#vehicle_state[:,0] = [thetay, 1000*x_pos[0], 1000*z_pos[0]]

# now integrate each motion estimate from the start
for i in range(frame_end-frame_start):
        theta, x, z = vehicle_state[:,i] # previous vehicle state
        #rot, trans = motion_estimates[:,i]  # estimated motion from previous
        #rot, tx, tz = motion_estimates[:,i]  # estimated motion from previous
        rotx, roty, rotz, tx, ty, tz = motion_estimates[:,i]  # estimated motion from previous
        trans = np.sqrt(tx ** 2 + tz ** 2)
        #theta = theta + rot # integrate
        theta = theta + roty # integrate
        x = x + trans *np.sin(theta) # integrate.
        z = z + trans *np.cos(theta) # 
        #x = x + tx
        #z = z + tz
        vehicle_state[:,i+1] = [theta, x, z]

plt.plot(x_pos,z_pos,'.', vehicle_state[1,:]/1000.0,vehicle_state[2,:]/1000.0,'+')
ax.set_title('Estimate vs Ground Truth Motion')
plt.show()
x_err =  (x_pos[-1] - vehicle_state[1,-1]/1000.0)**2
z_err = (z_pos[-1] - vehicle_state[2,-1]/1000.0)**2

print("Error in location at end is ", np.sqrt(x_err+z_err)," meters.")

x_err =  (x_pos - vehicle_state[1,:]/1000.0)**2
z_err = (z_pos - vehicle_state[2,:]/1000.0)**2
avg_err = np.sqrt( np.sum( x_err + z_err) )/(frame_end-frame_start+1)
print("Avg Sum Squared Error in location over run is", avg_err," meters.")