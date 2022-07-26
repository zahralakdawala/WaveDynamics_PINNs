# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:16:13 2022

@author: zahra
"""

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

## Functions

def f(x, y, t):
    return 0

def V(x, y):
    return 0

def I(x, y):
    return np.sin(x*np.pi/2 + y) 

def run(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2,
                       V=None, step1=False):
    if step1:
        dt = np.sqrt(dt2)  # save
        Cx2 = 0.5*Cx2;  Cy2 = 0.5*Cy2; dt2 = 0.5*dt2  # redefine
        D1 = 1;  D2 = 0
    else:
        D1 = 2;  D2 = 1
    u_xx = u_n[:-2,1:-1] - 2*u_n[1:-1,1:-1] + u_n[2:,1:-1]
    u_yy = u_n[1:-1,:-2] - 2*u_n[1:-1,1:-1] + u_n[1:-1,2:]
    u[1:-1,1:-1] = D1*u_n[1:-1,1:-1] - D2*u_nm1[1:-1,1:-1] + \
                   Cx2*u_xx + Cy2*u_yy + dt2*f_a[1:-1,1:-1]

    if step1:
        u[1:-1,1:-1] += dt*V[1:-1, 1:-1]

    # Boundary condition u=0
    j = 0
    u[:,j] = 0
    j = u.shape[1]-1
    u[:,j] = 0
    i = 0
    u[i,:] = 0
    i = u.shape[0]-1
    u[i,:] = 0

    return u


## Defining Variables

Nx = 100
Ny = 100

Lx = 10
Ly = 11

c = 0.5

dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(0, Lx, Nx+1)  # Mesh points in x dir
y = np.linspace(0, Ly, Ny+1)  # Mesh points in y dir

dt = 0.1
T  = 1
Nt = int(round(T/float(dt)))
t = np.linspace(0, Nt*dt, Nt+1)    # mesh points in time


xv = x[:,np.newaxis]          # For vectorized function evaluations
yv = y[np.newaxis,:]

stability_limit = (1/float(c))*(1/np.sqrt(1/dx**2 + 1/dy**2))

if dt <= 0:                # max time step?
    safety_factor = -dt    # use negative dt as safety factor
    dt = safety_factor*stability_limit

elif dt > stability_limit:
    print ('error: dt=%g exceeds the stability limit %g' %(dt, stability_limit))


Cx2 = (c*dt/dx)**2 
Cy2 = (c*dt/dy)**2    # help variables
dt2 = dt**2

 # Allow f and V to be None or 0
if f is None or f == 0:
    f = lambda x, y, t: np.zeros((x.shape[0], y.shape[1]))

if V is None or V == 0:
    V = lambda x, y: np.zeros((x.shape[0], y.shape[1]))

u     = np.zeros((Nx+1,Ny+1))   # Solution array
u_n   = np.zeros((Nx+1,Ny+1))   # Solution at t-dt
u_nm1 = np.zeros((Nx+1,Ny+1))   # Solution at t-2*dt
f_a   = np.zeros((Nx+1,Ny+1))   # For compiled loops


# Use vectorized version (requires I to be vectorized)
u_n[:,:] = I(xv, yv)

# Special formula for first time step
n = 0

f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
V_a = np.zeros((Nx+1, Ny+1))
V_a[:, :] = V(xv, yv)

u = run(u, u_n, u_nm1, f_a, Cx2,\
                       Cy2, dt2, V=V_a, step1=True)


# Update data structures for next step
#u_nm1[:] = u_n;  u_n[:] = u  # safe, but slow
u_nm1, u_n, u = u_n, u, u_nm1


for n in range(0, t.shape[0]):
    f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
    u = run(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2)

    # Update data structures for next step
    u_nm1, u_n, u = u_n, u, u_nm1

   # fig = plt.figure()
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, u_n,  cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(0.01)

