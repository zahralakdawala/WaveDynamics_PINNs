# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:15:21 2022

@author: zahra
"""

import numpy as np
import matplotlib.pyplot as plt

nx = 50
ny = 50

k = nx*ny

Lx = 1
Ly = 1

c = 0.5

T = 3
nt = 10
delta_t = T / nt


delta_x = Lx / nx
delta_y = Ly / ny

dx = np.square(c * delta_t / delta_x)
dy = np.square(c * delta_t / delta_y)

alpha = (1 + 2*dx + 2*dy)


# Coefficient matrix

A_diag = np.diag(np.ones((k,)) * alpha)
A_diag_upper_dx = np.diag(np.ones((k-1, )) * -dx, 1)
A_diag_upper_dy = np.diag(np.ones((k-nx, )) * -dy, nx)

for i in range(ny):
    A_diag_upper_dx[i*nx-1, i*nx] = 0

A_upper = A_diag_upper_dx + A_diag_upper_dy
A_lower = A_upper.T
A = A_upper + A_lower + A_diag


# Boundary Conditions

U = np.zeros((nx+2, ny+2))

U[0, :] = 0
U[-1, :] = 0
U[:, -1] = 0
U[:, 0] = 0

# Incooperating boundary conditions in b (Ax = b)

bound_vec = np.copy(U)

bound_vec_x = np.zeros(U.shape)

bound_vec_x[1, :] = bound_vec[0, :]
bound_vec_x[-2, :] = bound_vec[-1, :]

bound_vec_x = (bound_vec_x[1:-1, 1:-1]).reshape(-1, 1)

bound_vec_y = np.zeros(U.shape)

bound_vec_y[:, 1] = bound_vec[:, 0]
bound_vec_y[:, -2] = bound_vec[:, -1]

bound_vec_y = (bound_vec_y[1:-1, 1:-1]).reshape(-1, 1)


# Initial Conditions

x = np.linspace(0, Lx, nx+2).reshape(-1, 1)
y = np.linspace(0, Ly, ny+2).reshape(1, -1)

def I(x, y):
    return - np.sin(x + y) + np.cos(y)

In = I(x, y).reshape(nx+2, ny+2)

u_0 = np.copy(U)
u_0[1:-1, 1:-1] = In[1:-1, 1:-1]

# Using u_t = 0

u_1 = u_0

# b in Ax = b // Lets assume that boudaries are zero

u_0 = u_0[1:-1, 1:-1].reshape(-1, 1)
u_1 = u_1[1:-1, 1:-1].reshape(-1, 1)


b = 2*u_1 + u_0

for t in range(nt):
    u = np.linalg.solve(A, b)

    u_0, u_1 = u_1, u
    b = 2*u_1 + u_0

    U[1:-1, 1:-1] = u.reshape(nx, ny)   
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, U.T,  cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

