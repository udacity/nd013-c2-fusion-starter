# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate and plot results
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import colors
from matplotlib.transforms import Affine2D
import matplotlib.ticker as ticker
import os
import cv2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.waymo_reader.simple_waymo_open_dataset_reader import label_pb2
    
def plot_tracks(fig, ax, ax2, track_list, meas_list, lidar_labels, lidar_labels_valid, 
                      image, camera, configs_det, state=None):
    
    # plot image
    ax.cla()
    ax2.cla()
    ax2.imshow(image)
    
    # plot tracks, measurements and ground truth in birds-eye view
    for track in track_list:
        if state == None or track.state == state: # plot e.g. only confirmed tracks
            
            # choose color according to track state
            if track.state == 'confirmed':
                col = 'green'
            elif track.state == 'tentative':
                col = 'orange'
            else:
                col = 'red'
            
            # get current state variables    
            w = track.width
            h = track.height
            l = track.length
            x = track.x[0]
            y = track.x[1]
            z = track.x[2] 
            yaw = track.yaw 
                
            # plot boxes in top view
            point_of_rotation = np.array([w/2, l/2])        
            rec = plt.Rectangle(-point_of_rotation, width=w, height=l, 
                                    color=col, alpha=0.2,
                                    transform=Affine2D().rotate_around(*(0,0), -yaw)+Affine2D().translate(-y,x)+ax.transData)
            ax.add_patch(rec)
            
            # write track id for debugging
            ax.text(float(-track.x[1]), float(track.x[0]+1), str(track.id))
           
            if track.state =='initialized':
                ax.scatter(float(-track.x[1]), float(track.x[0]), color=col, s=80, marker='x', label='initialized track')
            elif track.state =='tentative':
                ax.scatter(float(-track.x[1]), float(track.x[0]), color=col, s=80, marker='x', label='tentative track')
            elif track.state =='confirmed':
                ax.scatter(float(-track.x[1]), float(track.x[0]), color=col, s=80, marker='x', label='confirmed track')
         
            # project tracks in image
            # transform from vehicle to camera coordinates
            pos_veh = np.ones((4, 1)) # homogeneous coordinates
            pos_veh[0:3] = track.x[0:3] 
            pos_sens = camera.veh_to_sens*pos_veh # transform from vehicle to sensor coordinates
            x = pos_sens[0]
            y = pos_sens[1]
            z = pos_sens[2] 
            
            # compute rotation around z axis
            R = np.matrix([[np.cos(yaw), np.sin(yaw), 0],
                        [-np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
            
            # bounding box corners
            x_corners = [-l/2, l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2]  
            y_corners = [-w/2, -w/2, -w/2, w/2, w/2, w/2, w/2, -w/2]  
            z_corners = [-h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2, h/2]  
            
            # bounding box
            corners_3D = np.array([x_corners, y_corners, z_corners])
            
            # rotate
            corners_3D = R*corners_3D

            # translate
            corners_3D += np.array([x, y, z]).reshape((3, 1))
            # print ( 'corners_3d', corners_3D)
            
            # remove bounding boxes that include negative x, projection makes no sense
            if np.any(corners_3D[0,:] <= 0):
                continue
            
            # project to image
            corners_2D = np.zeros((2,8))
            for k in range(8):
                corners_2D[0,k] = camera.c_i - camera.f_i * corners_3D[1,k] / corners_3D[0,k]
                corners_2D[1,k] = camera.c_j - camera.f_j * corners_3D[2,k] / corners_3D[0,k]
            # print ( 'corners_2d', corners_2D)

            # edges of bounding box in vertex index from above, e.g. index 0 stands for [-l/2, -w/2, -h/2]
            draw_line_indices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

            paths_2D = np.transpose(corners_2D[:, draw_line_indices])
            # print ( 'paths_2D', paths_2D)
            
            codes = [Path.LINETO]*paths_2D.shape[0]
            codes[0] = Path.MOVETO
            path = Path(paths_2D, codes)
                
            # plot bounding box in image
            p = patches.PathPatch(
                path, fill=False, color=col, linewidth=3)
            ax2.add_patch(p)
        
    # plot labels
    for label, valid in zip(lidar_labels, lidar_labels_valid):
        if valid:        
            ax.scatter(-1*label.box.center_y, label.box.center_x, color='gray', s=80, marker='+', label='ground truth')
    # plot measurements
    for meas in meas_list:
        ax.scatter(-1*meas.z[1], meas.z[0], color='blue', marker='.', label='measurement')
    
    # maximize window        
    mng = plt.get_current_fig_manager()
    mng.frame.Maximize(True)
    
    # axis 
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    ax.set_aspect('equal')
    ax.set_ylim(configs_det.lim_x[0], configs_det.lim_x[1]) # x forward, y left in vehicle coordinates
    ax.set_xlim(-configs_det.lim_y[1], -configs_det.lim_y[0])
    # correct x ticks (positive to the left)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(-x) if x!=0 else '{0:g}'.format(x))
    ax.xaxis.set_major_formatter(ticks_x)
    
    # remove repeated labels
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    ax.legend(handle_list, label_list, loc='center left', shadow=True, fontsize='x-large', bbox_to_anchor=(0.8, 0.5))

    plt.pause(0.01)
    
    return fig, ax, ax2


def plot_rmse(manager, all_labels, configs_det):
    fig, ax = plt.subplots()
    plot_empty = True
    
    # loop over all tracks
    for track_id in range(manager.last_id+1):
        rmse_sum = 0
        cnt = 0
        rmse = []
        time = []
        
        # loop over timesteps
        for i, result_dict in enumerate(manager.result_list):
            label_list = all_labels[i]
            if track_id not in result_dict:
                continue
            track = result_dict[track_id]
            if track.state != 'confirmed':
                continue
            
            # find closest label and calculate error at this timestamp
            min_error = np.inf
            for label, valid in zip(label_list[0], label_list[1]):
                error = 0
                if valid: 
                    # check if label lies inside specified range
                    if label.box.center_x > configs_det.lim_x[0] and label.box.center_x < configs_det.lim_x[1] and label.box.center_y > configs_det.lim_y[0] and label.box.center_y < configs_det.lim_y[1]:
                        error += (label.box.center_x - float(track.x[0]))**2
                        error += (label.box.center_y - float(track.x[1]))**2
                        error += (label.box.center_z - float(track.x[2]))**2
                        if error < min_error:
                            min_error = error
            if min_error < np.inf:
                error = np.sqrt(min_error)
                time.append(track.t)
                rmse.append(error)     
                rmse_sum += error
                cnt += 1
            
        # calc overall RMSE
        if cnt != 0:
            plot_empty = False
            rmse_sum /= cnt
            # plot RMSE
            ax.plot(time, rmse, marker='x', label='RMSE track ' + str(track_id) + '\n(mean: ' 
                    + '{:.2f}'.format(rmse_sum) + ')')
    
    # maximize window     
    mng = plt.get_current_fig_manager()
    mng.frame.Maximize(True)
    ax.set_ylim(0,1)
    if plot_empty: 
        print('No confirmed tracks found to plot RMSE!')
    else:
        plt.legend(loc='center left', shadow=True, fontsize='x-large', bbox_to_anchor=(0.9, 0.5))
        plt.xlabel('time [s]')
        plt.ylabel('RMSE [m]')
        plt.show()
        
        
def make_movie(path):
    # read track plots
    images = [img for img in sorted(os.listdir(path)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    # save with 10fps to result dir
    video = cv2.VideoWriter(os.path.join(path, 'my_tracking_results.avi'), 0, 10, (width,height))

    for image in images:
        fname = os.path.join(path, image)
        video.write(cv2.imread(fname))
        os.remove(fname) # clean up

    cv2.destroyAllWindows()
    video.release()