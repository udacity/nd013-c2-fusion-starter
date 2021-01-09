# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Parameter file for tracking
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general parameters
dim_state = 6 # process model dimension

# Kalman filter parameters (Step 1)
dt = 0.1 # time increment
q=3 # process noise variable for Kalman filter Q

# track management parameters (Step 2)
confirmed_threshold = 0.8 # track score threshold to switch from 'tentative' to 'confirmed'
delete_threshold = 0.6 # track score threshold to delete confirmed tracks
window = 6 # number of frames for track score calculation
max_P = 3**2 # delete track if covariance of px or py bigger than this
sigma_p44 = 50 # initial setting for estimation error covariance P entry for vx
sigma_p55 = 50 # initial setting for estimation error covariance P entry for vy
sigma_p66 = 5 # initial setting for estimation error covariance P entry for vz
weight_dim = 0.1 # sliding average parameter for dimension estimation

# association parameters (Step 3)
gating_threshold = 0.995 # percentage of correct measurements that shall lie inside gate

# measurement parameters (Step 4)
sigma_lidar_x = 0.1 # measurement noise standard deviation for lidar x position   
sigma_lidar_y = 0.1 # measurement noise standard deviation for lidar y position   
sigma_lidar_z = 0.1 # measurement noise standard deviation for lidar z position   
sigma_cam_i = 5 # measurement noise standard deviation for image i coordinate
sigma_cam_j = 5 # measurement noise standard deviation for image j coordinate
