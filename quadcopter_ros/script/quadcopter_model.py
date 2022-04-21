#!/usr/bin/env python3
from mimetypes import init
from glob import glob
import numpy as np
import control
import cvxpy as cp

class Quadcopter:
    def __init__(self, **init_kwargs):
        self.Ix = 1.
        self.Iy = 1.
        self.Iz = 1.5

        self.g = 9.8  
        self.m = 5.
        self.N = 20  
        self.DT = 0.1
        self.u0 = 10.
        self.reach_goal_thresh = 1.5
        self.nsim = 400  # Number of simulation time steps
        self.idx_incr = 18 # Amount of 'lookahead' for trajectory follower

        #  States
        self.x_dot = init_kwargs['x_dot'] if 'x_dot' in init_kwargs.keys() else 0.
        self.y_dot = init_kwargs['y_dot'] if 'y_dot' in init_kwargs.keys() else 0.
        self.z_dot = init_kwargs['z_dot'] if 'z_dot' in init_kwargs.keys() else 0.
        self.x = init_kwargs['x'] if 'x' in init_kwargs.keys() else 0.
        self.y = init_kwargs['y'] if 'y' in init_kwargs.keys() else 0.
        self.z = init_kwargs['z'] if 'z' in init_kwargs.keys() else 0.

        self.roll_dot = init_kwargs['roll_dot'] if 'roll_dot' in init_kwargs.keys() else 0.
        self.pitch_dot = init_kwargs['pitch_dot'] if 'pitch_dot' in init_kwargs.keys() else 0.
        self.yaw_dot = init_kwargs['yaw_dot'] if 'yaw_dot' in init_kwargs.keys() else 0.
        self.roll = init_kwargs['roll'] if 'roll' in init_kwargs.keys() else 0.
        self.pitch = init_kwargs['pitch'] if 'pitch' in init_kwargs.keys() else 0.
        self.yaw = init_kwargs['yaw'] if 'yaw' in init_kwargs.keys() else 0.

        self.A_zoh = np.eye(12)
        self.B_zoh = np.zeros((12, 4))

        self.states = np.array([self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot,
                                self.x_dot, self.y_dot, self.z_dot, self.x, self.y, self.z]).T

    @property
    def A(self):
        # Linear state transition matrix
        A = np.zeros((12, 12))
        A[0, 3] = 1.
        A[1, 4] = 1.
        A[2, 5] = 1.
        A[6, 1] = -self.g
        A[7, 0] = self.g
        A[9, 6] = 1.
        A[10, 7] = 1.
        A[11, 8] = 1.
        return A

    @property
    def B(self):
        # Control matrix
        B = np.zeros((12, 4))
        B[3, 1] = 1/self.Ix
        B[4, 2] = 1/self.Iy
        B[5, 3] = 1/self.Iz
        B[8, 0] = 1/self.m
        return B

    @property
    def C(self):
        C = np.eye(12)
        return C

    @property
    def D(self):
        D = np.zeros((12, 4))
        return D

    @property
    def Q(self):
        # State cost
        Q = np.eye(12)
        Q[8, 8] = 5.  # z vel
        Q[9, 9] = 10.  # x pos
        Q[10, 10] = 10.  # y pos
        Q[11, 11] = 100.  # z pos
        return Q

    @property
    def R(self):
        # Actuator cost
        R = np.eye(4)*.001
        return R

    def zoh(self):
        # Convert continuous time dynamics into discrete time
        sys = control.StateSpace(self.A, self.B, self.C, self.D)
        sys_discrete = control.c2d(sys, self.DT, method='zoh')

        self.A_zoh = np.array(sys_discrete.A)
        self.B_zoh = np.array(sys_discrete.B)

    def set_horizontal_length(self,N):
        self.N=N
        return self.N

    def run_mpc(self, rx,x,u,x_init):
        cost = 0.
        umin =np.array([7.7 , 7.7 , 7.7 , 7.7])-self.u0
        umax =np.array([13. , 13. , 13. , 13.])-self.u0
        INF = np.inf
        xmin = np.array([-0.2, -0.2, -2*np.pi, -.25, -.25, -.25,  -INF,  -INF,  -INF, -INF, -INF, -INF])
        xmax = np.array([0.2,  0.2,   2*np.pi,  .25, .25,  .25,   INF,   INF,   INF,   INF,  INF, INF])

        constr = [x[:, 0] == x_init]
        for t in range(self.N):
            cost += cp.quad_form(rx - x[:, t], self.Q) + cp.quad_form(u[:, t], self.R)  # Linear Quadratic cost
            constr += [xmin <= x[:, t], x[:, t] <= xmax]  # State constraints
            constr += [umin <= u[:, t], u[:, t] <= umax]
            constr += [x[:, t + 1] == self.A_zoh * x[:, t] + self.B_zoh * u[:, t]]

        cost += cp.quad_form(x[:, self.N] - rx, self.Q)  # End of trajectory error cost
        problem = cp.Problem(cp.Minimize(cost), constr)  
        return problem

    

    def update_states(self, ft, tx, ty, tz):
        roll_ddot =   tx / self.Ix
        pitch_ddot =  ty / self.Iy
        yaw_ddot =    tz / self.Iz
        x_ddot = -self.g*self.pitch
        y_ddot = self.g*self.roll
        z_ddot = -1*(self.g - (ft/self.m))

        self.roll_dot += roll_ddot*self.DT
        self.roll += self.roll_dot*self.DT
        self.pitch_dot += pitch_ddot * self.DT
        self.pitch += self.pitch_dot * self.DT
        self.yaw_dot += yaw_ddot * self.DT
        self.yaw += self.yaw_dot * self.DT

        self.x_dot += x_ddot * self.DT
        self.x += self.x_dot * self.DT
        self.y_dot += y_ddot * self.DT
        self.y += self.y_dot * self.DT
        self.z_dot += z_ddot * self.DT
        self.z += self.z_dot * self.DT

        self.states = np.array([self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot,
                                self.x_dot, self.y_dot, self.z_dot, self.x, self.y, self.z]).T



    def done(self,curr_pos, final_waypt):
        if np.linalg.norm(curr_pos - final_waypt) <= self.reach_goal_thresh:
            return True
        return False

    def __call__(self, ft=0., tx=0., ty=0., tz=0.):
        hover = self.m*9.8
        self.update_states(hover+ft, tx, ty, tz)