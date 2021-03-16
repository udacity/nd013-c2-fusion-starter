# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for sensor and measurement 
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Sensor:
    '''Sensor class including measurement matrix'''
    def __init__(self, name, calib):
        self.name = name
        if name == 'lidar':
            self.dim_meas = 3
            self.sens_to_veh = np.matrix(np.identity((4))) # transformation sensor to vehicle coordinates equals identity matrix because lidar detections are already in vehicle coordinates
            self.fov = [-np.pi/2, np.pi/2] # angle of field of view in radians
        
        elif name == 'camera':
            self.dim_meas = 2
            self.sens_to_veh = np.matrix(calib.extrinsic.transform).reshape(4,4) # transformation sensor to vehicle coordinates
            self.f_i = calib.intrinsic[0] # focal length i-coordinate
            self.f_j = calib.intrinsic[1] # focal length j-coordinate
            self.c_i = calib.intrinsic[2] # principal point i-coordinate
            self.c_j = calib.intrinsic[3] # principal point j-coordinate
            self.fov = [-0.35, 0.35] # angle of field of view in radians, inaccurate boundary region was removed
            
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh) # transformation vehicle to sensor coordinates
    
    def in_fov(self, x):
        # check if an object x can be seen by this sensor
        ############
        # TODO Step 4: implement a function that returns True if x lies in the sensor's field of view, 
        # otherwise False.
        ############

        return True
        
        ############
        # END student code
        ############ 
             
    def get_hx(self, x):    
        # calculate nonlinear measurement expectation value h(x)   
        if self.name == 'lidar':
            pos_veh = np.ones((4, 1)) # homogeneous coordinates
            pos_veh[0:3] = x[0:3] 
            pos_sens = self.veh_to_sens*pos_veh # transform from vehicle to lidar coordinates
            return pos_sens[0:3]
        elif self.name == 'camera':
            
            ############
            # TODO Step 4: implement nonlinear camera measurement function h:
            # - transform position estimate from vehicle to camera coordinates
            # - project from camera to image coordinates
            # - make sure to not divide by zero, raise an error if needed
            # - return h(x)
            ############

            pass
        
            ############
            # END student code
            ############ 
        
    def get_H(self, x):
        # calculate Jacobian H at current x from h(x)
        H = np.matrix(np.zeros((self.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3] # rotation
        T = self.veh_to_sens[0:3, 3] # translation
        if self.name == 'lidar':
            H[0:3, 0:3] = R
        elif self.name == 'camera':
            # check and print error message if dividing by zero
            if R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0] == 0: 
                raise NameError('Jacobian not defined for this x!')
            else:
                H[0,0] = self.f_i * (-R[1,0] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,0] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[1,0] = self.f_j * (-R[2,0] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,0] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[0,1] = self.f_i * (-R[1,1] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,1] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[1,1] = self.f_j * (-R[2,1] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,1] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[0,2] = self.f_i * (-R[1,2] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,2] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[1,2] = self.f_j * (-R[2,2] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,2] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
        return H   
        
    def generate_measurement(self, num_frame, z, meas_list):
        # generate new measurement from this sensor and add to measurement list
        ############
        # TODO Step 4: remove restriction to lidar in order to include camera as well
        ############
        
        if self.name == 'lidar':
            meas = Measurement(num_frame, z, self)
            meas_list.append(meas)
        return meas_list
        
        ############
        # END student code
        ############ 
        
        
################### 
        
class Measurement:
    '''Measurement class including measurement values, covariance, timestamp, sensor'''
    def __init__(self, num_frame, z, sensor):
        # create measurement object
        self.t = (num_frame - 1) * params.dt # time
        self.sensor = sensor # sensor that generated this measurement
        
        if sensor.name == 'lidar':
            sigma_lidar_x = params.sigma_lidar_x # load params
            sigma_lidar_y = params.sigma_lidar_y
            sigma_lidar_z = params.sigma_lidar_z
            self.z = np.zeros((sensor.dim_meas,1)) # measurement vector
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.z[2] = z[2]
            self.R = np.matrix([[sigma_lidar_x**2, 0, 0], # measurement noise covariance matrix
                                [0, sigma_lidar_y**2, 0], 
                                [0, 0, sigma_lidar_z**2]])
            
            self.width = z[4]
            self.length = z[5]
            self.height = z[3]
            self.yaw = z[6]
        elif sensor.name == 'camera':
            
            ############
            # TODO Step 4: initialize camera measurement including z and R 
            ############

            pass
        
            ############
            # END student code
            ############ 