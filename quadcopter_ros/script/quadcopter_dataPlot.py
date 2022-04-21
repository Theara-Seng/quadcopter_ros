#!/usr/bin/env python3
PKG ="quadcopter_ros"
from tkinter import E

from matplotlib import projections
import roslib; roslib.load_manifest(PKG)
from cv2 import SparsePyrLKOpticalFlow, destroyAllWindows, mulSpectrums
import airsim
from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import control
import cvxpy as cp
from math import sin as s, cos as c, atan2,asin,acos

from pandas import get_option
from script.splineGenerator import SplineGenerator
from script.quadcopter_model import Quadcopter
import rospy
from std_msgs.msg import Float64
from quadcopter_ros.msg import quad_traj,odometry,spline_traj,goal_found,time_exe,input_msg

class dataplot_ros(Quadcopter):
    def __init__(self):
        
        rospy.init_node("dataplot_ros")
        self.odom_data_sub=rospy.Subscriber("odom_publisher",odometry,self.odom_data_sub_callback)
        self.spline_data_sub=rospy.Subscriber("spline_traj",spline_traj,self.spline_data_sub_callback)
        self.goal_found_sub=rospy.Subscriber("goal_found",goal_found,self.goal_found_callback)
        self.input_sub=rospy.Subscriber("input_quad",input_msg,self.input_quad_callback)

        self.odom=odometry()
        self.spline=spline_traj()
        
        self.time_lenght=[]
        self.time_input=[]
        self.goal_found=False

        self.x_pos=[]
        self.y_pos=[]
        self.z_pos=[]
        self.x_vel=[]
        self.y_vel=[]
        self.z_vel=[]
        self.roll=[]
        self.pitch=[]
        self.yaw=[]
        self.rollRate=[]
        self.pitchRate=[]
        self.yawRate=[]

        self.spline_x_data=[]
        self.spline_y_data=[]
        self.spline_z_data=[]
        self.waypoint=[]

        self.thrust=[]
        self.x_torque=[]
        self.y_torque=[]
        self.z_torque=[]

    def input_quad_callback(self,msg):
        self.thrust.append(msg.thrust)
        self.x_torque.append(msg.x_torque)
        self.y_torque.append(msg.y_torque)
        self.z_torque.append(msg.z_torque)
        time_input=len(self.thrust)
        time_input=time_input*0.1
        self.time_input.append(time_input)

    def odom_data_sub_callback(self,odom):
        self.x_pos.append(odom.pose[0])
        self.y_pos.append(odom.pose[1])
        self.z_pos.append(odom.pose[2])

        self.x_vel.append(odom.vel[0])
        self.y_vel.append(odom.vel[1])
        self.z_vel.append(odom.vel[2])
        
        self.roll.append(odom.ang[0])
        self.pitch.append(odom.ang[1])
        self.yaw.append(odom.ang[2])
        
        self.rollRate.append(odom.angRate[0])
        self.pitchRate.append(odom.angRate[1])
        self.yawRate.append(odom.angRate[2])

        time_lenght=len(self.x_pos)
        time_lenght=time_lenght*self.DT
        self.time_lenght.append(time_lenght)
    def spline_data_sub_callback(self,spline):
        self.spline_x_data=spline.x_data
        self.spline_y_data=spline.y_data
        self.spline_z_data=spline.z_data
        self.waypoint=spline.way_points

    def goal_found_callback(self,msg):
        self.goal_found=msg.goal_found

    def plot_trajectory_3d(self):
        if self.goal_found==True:
            fig  = plt.figure(figsize=(15,15))
            ax = p3.Axes3D(fig)
            ax.set_xlim(-10., 10.)
            ax.set_ylim(-10., 10.)
            ax.set_zlim(-10., 10.)

            writer = animation.writers['ffmpeg']
            writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

            def update(ii):
              
                quad_pos.set_data(self.x_pos[ii], self.y_pos[ii])
                quad_pos.set_3d_properties(self.z_pos[ii])

                spline.set_data(self.spline_x_data, self.spline_y_data)
                spline.set_3d_properties(self.spline_z_data)
                return (quad_pos,) + (spline,) 


            quad_pos, = plt.plot([], [], 'bX', markersize=10.)
            spline, = plt.plot([], [], 'r-', markersize=5.)
            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
           
            line_ani = animation.FuncAnimation(fig, update, len(self.x_pos), interval=100, repeat=True)
            plt.show()

    def plot_data(self):
        if self.goal_found==True:
           
            plt.subplot(221)
            waypoint=np.asarray(self.waypoint)
            waypoint=waypoint.reshape(-1,3)
            plt.plot(self.x_pos, self.y_pos, 'b-', label='Quadcopter')
            plt.plot(self.spline_x_data, self.spline_y_data, 'r-', label='Trajectory')
            plt.plot(waypoint[:, 0], waypoint[:, 1], 'gx', label='Waypoints')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.grid()
            plt.legend()
            plt.title('Trajectory')
            
            plt.subplot(222)
            plt.plot(self.time_lenght, self.x_pos, label='x')
            plt.plot(self.time_lenght, self.y_pos, label='y')
            plt.plot(self.time_lenght, self.z_pos, label='z')
            plt.plot(self.time_lenght, self.x_vel, label='x vel')
            plt.plot(self.time_lenght, self.y_vel, label='y vel')
            plt.plot(self.time_lenght, self.z_vel, label='z vel')
            plt.xlabel('Time (s)')
            plt.ylabel('Linear State')
            plt.grid()
            plt.legend()
            plt.title('Linear States')
        

            plt.subplot(223)
            plt.plot(self.time_lenght, self.roll, label='roll')
            plt.plot(self.time_lenght, self.pitch, label='pitch')
            plt.plot(self.time_lenght, self.yaw, label='yaw')
            plt.plot(self.time_lenght, self.rollRate, label='roll rate')
            plt.plot(self.time_lenght, self.pitchRate, label='pitch rate')
            plt.plot(self.time_lenght, self.yawRate, label='yaw rate')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.legend()
            plt.title('Angular States')

            plt.subplot(224)
            plt.plot(self.time_input, self.thrust, label='ft')
            plt.plot(self.time_input, self.x_torque, label='tx')
            plt.plot(self.time_input, self.y_torque, label='ty')
            plt.plot(self.time_input, self.z_torque, label='tz')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.legend()
            plt.title('Control Inputs')

            plt.show()
            
        else:
            print("goal_found==False")


if __name__ == "__main__":
    dataplot=dataplot_ros()

    while not rospy.is_shutdown():
        dataplot.plot_trajectory_3d()
        dataplot.plot_data()
    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("dataplot keyboard interrupt")