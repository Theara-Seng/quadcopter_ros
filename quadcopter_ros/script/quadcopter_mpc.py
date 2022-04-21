#!/usr/bin/env python3
PKG ="quadcopter_ros"
from tkinter import E
from turtle import position
import roslib; roslib.load_manifest(PKG)
from cv2 import SparsePyrLKOpticalFlow, destroyAllWindows, fastNlMeansDenoising, mulSpectrums
import airsim
from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import control
import cvxpy as cp
from math import sin as s, cos as c, atan2,asin,acos
import argparse
import sys

from pandas import get_option
from script.splineGenerator import SplineGenerator

from script.quadcopter_model import Quadcopter
import rospy
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from quadcopter_ros.msg import quad_traj,odometry,spline_traj,goal_found,time_exe,input_msg
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time

class mpc_controllers(Quadcopter):
    def __init__(self,waypoints):
        super(mpc_controllers,self).__init__()
        rospy.init_node("trajectory_publisher")
        
        self.odom_data_sub=rospy.Subscriber("odom_publisher", odometry , self.odom_data_callback)
        self.odom_ned_sub=rospy.Subscriber("/airsim_node/drone_vb/odom_local_ned", Odometry , self.odometry_callback)

        self.traj_pub=rospy.Publisher("traj_publisher" , quad_traj,queue_size=10)
        self.goal_pub=rospy.Publisher("goal_found" , goal_found,queue_size=5)
        self.input_pub=rospy.Publisher("input_quad" , input_msg,queue_size=10)
        self.spline_data_pub=rospy.Publisher("spline_traj" , spline_traj,queue_size=10)

        self.odometry = Odometry()
        self.splines = spline_traj()
        self.goal_found = goal_found()
        self.goal_found.goal_found = False
        self.goal_reach = False
        self.waypoints=waypoints
    
    def odometry_callback(self,odom):
        self.odometry=odom.pose.pose.position
        
    def odom_data_callback(self,init_odom):
        
        init_pos=init_odom.pose
        init_vel =init_odom.vel
        init_ang =init_odom.ang
        init_angRate=init_odom.angRate

        init_dict = {'roll':init_ang[0], 'pitch':init_ang[1], 'yaw':init_ang[2],
                    'roll_dot': init_angRate[0], 'pitch_dot': init_angRate[1], 'yaw_dot': init_angRate[2],
                    'x_dot': init_vel[0],'y_dot': init_vel[1], 'z_dot':init_vel[2], 
                    'x': init_pos[0], 'y': init_pos[1], 'z': init_pos[2]}

        init_pose=np.array([init_pos[0],init_pos[1],init_pos[2]])
    
        quad = Quadcopter(**init_dict)
        quad.zoh()
    
        

        spline_gen = SplineGenerator()
        spline_data = spline_gen.create_splines(self.waypoints,init_pose)
        
        spline_x_data = spline_data[:, 0]
        spline_y_data = spline_data[:, 1]
        spline_z_data = spline_data[:, 2]

        way_points=[]
    
        for row in self.waypoints:
            for ele in row:
                way_points.append(ele)

        self.splines.way_points=way_points
        self.splines.x_data=spline_x_data
        self.splines.y_data=spline_y_data
        self.splines.z_data=spline_z_data
        self.spline_data_pub.publish(self.splines)

        # Initial solver states (copy of quadcopter states)
        x0 = np.array([init_dict['roll'], init_dict['pitch'], init_dict['yaw'], init_dict['roll_dot'],
                    init_dict['pitch_dot'], init_dict['yaw_dot'], init_dict['x_dot'], init_dict['y_dot'],
                    init_dict['x_dot'], init_dict['x'], init_dict['y'], init_dict['z']])

        # Desired states to track
        des_states = {'roll': 0., 'pitch': 0., 'yaw':init_ang[2], 'roll_dot': 0., 'pitch_dot': 0., 'yaw_dot': 0., 'x_dot': 0.,
                    'y_dot': 0., 'z_dot': 0., 'x': self.waypoints[0][0], 'y':self.waypoints[0][1], 'z':self.waypoints[0][2]}

        
        idx = self.idx_incr  # Index for current spline point

        [nx, nu] = quad.B.shape
            
        # Convex optimization solver variables
        x = cp.Variable((nx, self.N+1))
        u = cp.Variable((nu, self.N))
        x_init = cp.Parameter(nx)
    
        
        for i in range(1,self.nsim+1):
            
            ref_x = spline_x_data[idx]
            ref_y = spline_y_data[idx]
            ref_z = spline_z_data[idx]

            idx += self.idx_incr

            # If spline index tries to go past last waypoint
            if idx >= len(spline_x_data):
                idx = len(spline_x_data) - 1

            # Update reference states
            xr = np.array([des_states['roll'], des_states['pitch'], des_states['yaw'], des_states['roll_dot'],
                        des_states['pitch_dot'], des_states['yaw_dot'], des_states['x_dot'], des_states['y_dot'],
                        des_states['x_dot'], ref_x, ref_y, ref_z])

            # Run optimization for N horizons
            prob = quad.run_mpc(xr,x,u,x_init)

            # Solve convex optimization problem
            x_init.value=x0
            prob.solve(solver=cp.OSQP, warm_start=True)
            x0 = quad.A_zoh.dot(x0) + quad.B_zoh.dot(u[:, 0].value)

            # Send only first calculated command to quadcopter, then run optimization again
            quad(u[0, 0].value, u[1, 0].value, u[2, 0].value, u[3, 0].value)

            #input
            in_msg=input_msg()
            in_msg.thrust=u[0,0].value+(self.m*self.g)
            in_msg.x_torque=u[1,0].value
            in_msg.y_torque=u[2,0].value
            in_msg.z_torque=u[3,0].value
            self.input_pub.publish(in_msg)
            
            #output
            q=quad_traj()
            q.quad_x=quad.x
            q.quad_y=quad.y
            q.quad_z=quad.z
            self.traj_pub.publish(q)

            if quad.done(np.array([self.odometry.x,self.odometry.y,self.odometry.z]),np.array([self.waypoints[-1][0], self.waypoints[-1][1],self.waypoints[-1][2]])):
                print('GOAL REACHED')               
                self.goal_reach = True
                self.goal_found.goal_found=self.goal_reach
                self.goal_pub.publish(self.goal_reach)
                break

        if not self.goal_reach:
            print("GOAL FAILED")
            self.goal_reach=False
            self.goal_found.goal_found=self.goal_reach
            self.goal_pub.publish(self.goal_reach)
            

    

if __name__=="__main__":
  
    if rospy.has_param("/waypoints"):
        data=rospy.get_param("/waypoints")
        waypoints=[float(item) for item in data]
        dimension=3
        waypoints=np.asarray(waypoints).reshape((len(waypoints)//dimension),dimension)
    else:
        waypoints = np.array([[-4,-5,-6],[-3,-3,-8],[0,3,-5],[0,10,-7]])

    mpc_controllers=mpc_controllers(waypoints)

    rate=rospy.Rate(10)
    
    while not rospy.is_shutdown():
        rate.sleep()
 

    try:
        rospy.spin()         
    except KeyboardInterrupt:
        print("interrup mpc")
   


    
  
            
       
