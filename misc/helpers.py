# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : helper functions for loop_over_dataset.py
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import os
import pickle

## Saves an object to a binary file
def save_object_to_file(object, file_path, base_filename, object_name, frame_id=1):
    object_filename = os.path.join(file_path, os.path.splitext(base_filename)[0]
                                   + "__frame-" + str(frame_id) + "__" + object_name + ".pkl")
    with open(object_filename, 'wb') as f:
        pickle.dump(object, f)

## Loads an object from a binary file
def load_object_from_file(file_path, base_filename, object_name, frame_id=1):
    object_filename = os.path.join(file_path, os.path.splitext(base_filename)[0]
                                   + "__frame-" + str(frame_id) + "__" + object_name + ".pkl")
    with open(object_filename, 'rb') as f:
        object = pickle.load(f)
        return object
    
## Prepares an exec_list with all tasks to be executed
def make_exec_list(exec_detection, exec_tracking, exec_visualization): 
    
    # save all tasks in exec_list
    exec_list = exec_detection + exec_tracking + exec_visualization
    
    # check if we need pcl
    if any(i in exec_list for i in ('validate_object_labels', 'bev_from_pcl')):
        exec_list.append('pcl_from_rangeimage')
    # check if we need image
    if any(i in exec_list for i in ('show_tracks', 'show_labels_in_image', 'show_objects_in_bev_labels_in_camera')):
        exec_list.append('load_image')
    # movie does not work without show_tracks
    if 'make_tracking_movie' in exec_list:  
        exec_list.append('show_tracks')  
    return exec_list