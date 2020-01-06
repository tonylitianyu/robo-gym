import gym
from gym import error, spaces, utils

from gym.utils import seeding

import numpy as np
import math
import scipy.integrate
import sympy as sp

from vpython import box, sphere, color, vector, rate, canvas, cylinder, arrow, curve, compound,label

import controlpy
from random import random

FPS = 200

class QuadrotorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):


        self.gravity = 9.81
        self.drone = None
        self.state = np.zeros(12)
        self.input = np.zeros(4)

        #visualize the euler angle in simualtor
        self.xPointer = None
        self.yPointer = None
        self.zPointer = None
        self.x_axis = None
        self.y_axis = None
        self.z_axis = None


        #time
        self.t = 0
        self.dt = 0.02

        #drone moment of inertia property
        self.motor_mass = 1;
        self.drone_mass = 4*self.motor_mass;
        self.beam_length = 5
        xz_dir_length = (self.beam_length/2)*math.cos(45)
        self.Ix = 4*(self.motor_mass*(xz_dir_length**2))
        self.Iy = 4*(self.motor_mass*2*(xz_dir_length**2))
        self.Iz = 4*(self.motor_mass*(xz_dir_length**2))
        self.a1 = (self.Iy - self.Iz)/self.Ix
        self.a2 = (self.Iz - self.Ix)/self.Iy
        self.a3 = (self.Ix - self.Iy)/self.Iz

        #define limiting space
        self.max_bank_angle = 0.5
        high = np.array([50,50,50,50,50,50,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([30,-200,-200,-200]),high=np.array([50,200,200,200]),dtype=np.float32)

        self.initRender()

    def step(self, action):
        self.input = action
        # X_goal = np.array([10,0,10,0,10,0,0,0,0,0,0,0])
        # error = self.state-X_goal
        # error = np.reshape(error,(12,1))
        #
        # self.input = -self.LQRTest()@error
        # self.input[0] = self.limitTorque(self.input[0],'t')
        # self.input[2] = self.limitTorque(self.input[2],'x')
        self.input[3] = self.input[3] + 35
        state_augmented = np.append(self.state, self.input)
        sol = scipy.integrate.solve_ivp(self._dsdt, [0, self.dt], state_augmented)

        ns = sol.y[:,-1]
        self.state = ns[:-4]

        phi_   = self.state[6]
        theta_ = self.state[7]
        psi_   = self.state[8]
        self.state[6] = self.bound(phi_,-self.max_bank_angle,self.max_bank_angle)
        self.state[7] = self.bound(theta_,-self.max_bank_angle,self.max_bank_angle)
        self.state[8] = self.bound(psi_,-self.max_bank_angle,self.max_bank_angle)

        reward = 0
        done = False
        #if in any of the direction, the drone is going out of bounds, terminate this episode
        if max(np.absolute(self.state[[0,2,4]])) > 50:
            done = True
        return self.state,done


    def reset(self):
        self.state = np.array([-5+10*random(),0,-5+10*random(),0,0+20*random(),0,0,0,0,0,0,0],dtype=np.float32)
        self.drone.pos = vector(0,0,0)
        self.drone.axis = vector(1,0,0)
        self.drone.up = vector(0,1,0)
        self.xPointer.pos = self.drone.pos
        self.yPointer.pos = self.drone.pos
        self.zPointer.pos = self.drone.pos

        self.yPointer.axis = 7*self.drone.axis
        self.zPointer.axis = 7*self.drone.up
        xaxis = self.drone.axis.cross(self.zPointer.axis)
        self.xPointer.axis = xaxis
        return self.state
    def initRender(self):
        self.canvas = canvas(width=1200, height=900, title='Quadrotor-3D')
        ground_y = -0.5
        thk = 0.5
        ground_width = 200
        wallB = box(canvas=self.canvas,pos=vector(0, ground_y, 0), size=vector(ground_width, thk, ground_width),  color = vector(0.9,0.9,0.9))


        beam_length = self.beam_length
        beam_radius = 0.4
        beam1 = cylinder(pos=vector(-beam_length/2,0,-beam_length/2),axis=vector(beam_length,0,beam_length), radius=beam_radius,color=vector(0.3,0.3,0.3))
        beam2 = cylinder(pos=vector(-beam_length/2,0,beam_length/2),axis=vector(beam_length,0,-beam_length), radius=beam_radius,color=vector(0.3,0.3,0.3))

        propeller_height = 0.4
        propeller_radius = 1.5
        propeller_y = 0.4
        propeller1 = cylinder(pos=vector(-beam_length/2,propeller_y,-beam_length/2),axis=vector(0,propeller_height,0), radius=propeller_radius,color=color.green)
        propeller2 = cylinder(pos=vector(beam_length/2,propeller_y,-beam_length/2),axis=vector(0,propeller_height,0), radius=propeller_radius,color=color.red)
        propeller3 = cylinder(pos=vector(-beam_length/2,propeller_y,beam_length/2),axis=vector(0,propeller_height,0), radius=propeller_radius,color=color.purple)
        propeller4 = cylinder(pos=vector(beam_length/2,propeller_y,beam_length/2),axis=vector(0,propeller_height,0), radius=propeller_radius,color=color.cyan)
        self.drone = compound([beam1, beam2, propeller1, propeller2, propeller3, propeller4],pos = vector(0,0,0),make_trail=False,retain=300)

        #yzx

        self.xPointer = arrow(pos=self.drone.pos,axis=vector(0,0,-7),shaftwidth=0.3,color=color.red)
        self.yPointer = arrow(pos=self.drone.pos,axis=self.drone.axis,shaftwidth=0.3,color=color.blue)
        self.zPointer = arrow(pos=self.drone.pos,axis=7*self.drone.up,shaftwidth=0.3,color=color.green)
        self.drone.mass = self.drone_mass
        origin = sphere(pos=vector(0,0,0), radius=0.5, color=color.yellow)

        self.x_axis = arrow(pos=vector(0,0,0),axis=vector(0,0,3),shaftwidth=0.3,color=color.red)
        self.y_axis = arrow(pos=vector(0,0,0),axis=vector(3,0,0),shaftwidth=0.3,color=color.blue)
        self.z_axis = arrow(pos=vector(0,0,0),axis=vector(0,3,0),shaftwidth=0.3,color=color.green)
        rate(FPS)

    def render(self, mode='human', close=False):
        #self.t += 0.001
        phi_   = self.state[6]
        theta_ = self.state[7]
        psi_   = self.state[8]
        #print(self.state)
        self.drone.pos = vector(self.state[2],self.state[4],self.state[0])
        self.drone.up = vector(0,1,0)#vector(-math.sin(phi_),math.cos(phi_)+math.cos(theta_),math.sin(theta_))
        self.drone.axis = vector(1,0,0)#vector(math.cos(psi_),(math.sin(phi_)*math.cos(psi_)-math.sin(theta_)*math.sin(psi_))/(math.cos(phi_)*math.cos(theta_)),math.sin(psi_))
        self.drone.rotate(angle=phi_,axis=self.x_axis.axis)
        self.drone.rotate(angle=theta_,axis=self.y_axis.axis)
        self.drone.rotate(angle=psi_,axis=self.z_axis.axis)

        self.xPointer.pos = self.drone.pos
        self.yPointer.pos = self.drone.pos
        self.zPointer.pos = self.drone.pos

        self.yPointer.axis = 7*self.drone.axis
        self.zPointer.axis = 7*self.drone.up
        xaxis = self.drone.axis.cross(self.zPointer.axis)
        self.xPointer.axis = xaxis

        return True




    def _dsdt(self,t, s_augmented):

        #Outer Loop states
        x1 = s_augmented[0]
        x2 = s_augmented[1]#vel
        y1 = s_augmented[2]
        y2 = s_augmented[3]#vel
        z1 = s_augmented[4]
        z2 = s_augmented[5]#vel
        # Innner Loop States
        phi = s_augmented[6]
        theta = s_augmented[7]
        psi = s_augmented[8]
        p = s_augmented[9]
        q = s_augmented[10]
        r = s_augmented[11]
        # Control
        U1 = s_augmented[12]
        U2 = s_augmented[13]#rotation
        U3 = s_augmented[14]#rotation
        U4 = s_augmented[15]#rotation

        x1Dot = x2
        x2Dot = (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi))*(U1/self.drone_mass)
        y1Dot = y2
        y2Dot = (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))*(U1/self.drone_mass)
        z1Dot = z2
        z2Dot = -self.gravity + np.cos(phi)*np.cos(theta)*(U1/self.drone_mass)


        phiDot = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
        thetaDot = q*np.cos(phi) - r*np.sin(phi)
        psiDot = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)


        pDot = self.a1*r*q + float(U2/self.Ix)
        qDot = self.a2*r*p + float(U3/self.Iy)
        rDot = self.a3*p*q + float(U4/self.Iz)


        return x1Dot, x2Dot, y1Dot, y2Dot, z1Dot, z2Dot, phiDot, thetaDot, psiDot, pDot, qDot, rDot, 0.,0.,0.,0.





    def LQRTest(self):
        x1 = sp.Symbol('x1')
        x2 = sp.Symbol('x2')#vel
        y1 = sp.Symbol('y1')
        y2 = sp.Symbol('y2')#vel
        z1 = sp.Symbol('z1')
        z2 = sp.Symbol('z2')#vel

        phi = sp.Symbol('phi')
        theta = sp.Symbol('theta')
        psi = sp.Symbol('psi')
        p = sp.Symbol('p')
        q = sp.Symbol('q')
        r = sp.Symbol('r')

        U1 = sp.Symbol('U1')
        U2 = sp.Symbol('U2')
        U3 = sp.Symbol('U3')
        U4 = sp.Symbol('U4')



        X = np.array([x1,x2,y1,y2,z1,z2,phi,theta,psi,p,q,r])
        U = np.array([U1,U2,U3,U4])


        #X_err = self.state-X_goal


        x1Dot = x2
        x2Dot = ((sp.cos(phi)*sp.sin(theta)*sp.cos(psi) + sp.sin(phi)*sp.sin(psi))*(U1/self.drone_mass))
        y1Dot = y2
        y2Dot = ((sp.cos(phi)*sp.sin(theta)*sp.sin(psi) - sp.sin(phi)*sp.cos(psi))*(U1/self.drone_mass))
        z1Dot = z2

        z2Dot = -self.gravity + sp.cos(phi)*sp.cos(theta)*(U1/self.drone_mass)


        phiDot = p + q*sp.sin(phi)*sp.tan(theta) + r*sp.cos(phi)*sp.tan(theta)
        thetaDot = q*sp.cos(phi) - r*sp.sin(phi)
        psiDot = q*sp.sin(phi)/sp.cos(theta) + r*sp.cos(phi)/sp.cos(theta)


        pDot = self.a1*r*q + U2/self.Ix
        qDot = self.a2*r*p + U3/self.Iy
        rDot = self.a3*p*q + U4/self.Iz

        A = sp.Matrix([x1Dot,x2Dot,y1Dot,y2Dot,z1Dot,z2Dot,phiDot,thetaDot,psiDot,pDot,qDot,rDot])
        A = A.jacobian(X)
        B = sp.Matrix([x1Dot,x2Dot,y1Dot,y2Dot,z1Dot,z2Dot,phiDot,thetaDot,psiDot,pDot,qDot,rDot])
        B = B.jacobian(U)
        A = A.subs({x1:0,x2:0,y1:0,y2:0,z1:0,z2:0,phi:0,theta:0,psi:0,p:0,q:0,r:0,U1:self.gravity*self.drone_mass})

        B = B.subs({x1:0,x2:0,y1:0,y2:0,z1:0,z2:0,phi:0,theta:0,psi:0,p:0,q:0,r:0,U1:self.gravity*self.drone_mass})

        K = np.ones((4,12))

        A = np.array(A).astype(float)
        B = np.array(B).astype(float)

        # Q = np.eye(12)*1000
        R = np.eye(4)
        Q = np.diag([0.5,1,0.5,1,1,1,1,1,1,100,100,100])*200


        K = controlpy.synthesis.controller_lqr(A,B,Q,R)
        K = np.array(K[0])

        d,v = np.linalg.eig(A-B@K)

        return K



    def bound(self,x,m,M):
        return min(max(x, m), M)

    def limitTorque(self,torque,axis):
        #print('torque: '+str(torque)+' on ' + axis)
        if axis == 'x':
            return min(torque,300)
        else:
            return min(torque,300)
