#!/usr/bin/env python3
import numpy as np
from math import sin as s, cos as c, atan2,asin,acos

class SplineGenerator:
    def __init__(self):
        self.waypoints = None
        self.spline_pts = None
  
    def create_splines(self, waypts,initial_pos):
      
        self.waypoints = waypts.reshape((-1, 3))
        waypts = np.insert(waypts, 0, initial_pos,axis=0)
        waypts = np.insert(waypts, 0, initial_pos, axis=0)
        waypts = np.insert(waypts, -1, waypts[-1], axis=0)
        splines = []
        for j in range(len(waypts)-3):
            this_spline = self.cubic_spline(waypts[j], waypts[j+1], waypts[j+2], waypts[j+3])
            splines.append(this_spline)

        self.spline_pts = np.array(splines).reshape((-1, 4))
        return self.spline_pts

    def create_3d_traj(self,waypts,initial_pose):
        self.waypoints=waypts.reshape((-1,3))
        waypts=np.insert(waypts,0,initial_pose,axis=0)
        spline=[]
        for i in range(len(waypts)-1):
            this_spline=np.linspace(waypts[i],waypts[i+1],num=500)
            spline.append(this_spline)
        self.spline_pts=np.array(spline).reshape(-1,3)
        return self.spline_pts
   
    @staticmethod
    def cubic_spline(y0, y1, y2, y3, delt_mu=.001):
        mu = 0.
        points = []
        prev_x = 0.
        prev_y = 0.
        prev_z =0
        while mu <= 1.:
            mu2 = mu*mu
            a0 = y3 - y2 - y0 + y1
            a1 = y0 - y1 - a0
            a2 = y2 - y0
            a3 = y1
            mu += delt_mu
            point = a0*mu*mu2+a1*mu2+a2*mu+a3
            slope = atan2(point[1]-prev_y, point[0]-prev_x)
            point = np.append(point, slope)
            points.append(point)
            prev_x = point[0]
            prev_y = point[1]
            prev_z = point[2]

        return points

