#!/usr/bin/env python3
from ast import get_docstring
from curses import noecho
from turtle import position
import airsim
from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import control
import cvxpy as cp
from math import sin as s, cos as c, atan2,asin,acos
from script.splineGenerator import SplineGenerator

from script.quadcopter_model import Quadcopter
import rospy
from std_msgs.msg import Float64
from quadcopter_ros.msg import quad_traj,odometry,goal_found,spline_traj,input_msg
import time

class odom_quadcopter:

    def __init__(self,drone_name):

        self.drone_name=drone_name
        rospy.init_node("odometry_publisher",anonymous=False)

        self.odom_pub = rospy.Publisher("odom_publisher",odometry,queue_size=10)

        self.traj_sub = rospy.Subscriber("traj_publisher",quad_traj,self.quad_traj_callback)
        self.goal_sub = rospy.Subscriber("goal_found",goal_found,self.goal_found_callback)

        self.odometry = odometry()
        self.quad = quad_traj()
      

        self.quad.quad_x = None
        self.quad.quad_y = None
        self.quad.quad_z = None


        self.goal_found = False

        self.client = airsim.MultirotorClient()
        self.odom   = airsim.MultirotorClient()
        self.odom.confirmConnection()
        self.client.confirmConnection()
        self.client.enableApiControl(True,vehicle_name=self.drone_name)
        self.client.armDisarm(True,vehicle_name=self.drone_name)

        self.pose = 0.0
        self.vel = 0.0
        self.ang = 0.0
        self.angvel = 0.0

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0




    def takeoff_quadcopter(self):
        self.client.takeoffAsync(5,vehicle_name=self.drone_name).join()  
        
    def get_odom_state(self):
        drone_state = self.odom.getMultirotorState(vehicle_name=self.drone_name)
        self.pos   = drone_state.kinematics_estimated.position
        self.vel   = drone_state.kinematics_estimated.linear_velocity
        self.ang   = drone_state.kinematics_estimated.orientation  
        self.angvel= drone_state.kinematics_estimated.angular_velocity
        self.roll,self.pitch,self.yaw=self.euler_from_quaternion(self.ang.x_val,self.ang.y_val,self.ang.z_val,self.ang.z_val)
        self.odometry.pose=[self.pos.x_val,self.pos.y_val,self.pos.z_val]
        self.odometry.vel =[self.vel.x_val,self.vel.y_val,self.vel.z_val]
        self.odometry.ang =[self.roll,self.pitch,self.yaw]
        self.odometry.angRate=[self.angvel.x_val,self.angvel.y_val,self.angvel.z_val]
       
    def pub_odom_data(self):
        self.odom_pub.publish(self.odometry)

    def euler_from_quaternion(self,x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = atan2(t3, t4)

        return roll, pitch, yaw

    def goal_found_callback(self,msg):
        self.goal_found=msg.goal_found

    def quad_traj_callback(self,quads):
        
        self.quad.quad_x=quads.quad_x
        self.quad.quad_y=quads.quad_y
        self.quad.quad_z=quads.quad_z
       
    def move_to_target_pos(self):
        if self.goal_found==False:
            self.client.moveToPositionAsync(self.quad.quad_x,self.quad.quad_y,self.quad.quad_z,1.2,0.1,vehicle_name=self.drone_name)
        else:
            time.sleep(2)
            self.client.hoverAsync(vehicle_name=self.drone_name)
            self.client.reset()
            rospy.spin()
            
def main():
    odom_quad=odom_quadcopter(drone_name="drone_vb")

    odom_quad.takeoff_quadcopter()
    
    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        odom_quad.get_odom_state()
        odom_quad.pub_odom_data()
        odom_quad.move_to_target_pos()
        rate.sleep()
    
if __name__== "__main__":

    main()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("odom interrupt")