#!/usr/bin/env python3

import rospy

import airsim
import sys
import time


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)


