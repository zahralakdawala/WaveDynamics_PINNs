# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:17:17 2022

@author: zahra
"""

import numpy as np
import matplotlib.pyplot as plt

nx = 10
ny = 10
nz = 10

k = nx*ny*nz

Lx = 1
Ly = 1
Lz = 0.1

c = 0.5

T = 3
nt = 10
delta_t = T / nt


delta_x = Lx / nx
delta_y = Ly / ny
delta_z = Lz / nz

dx = np.square(c * delta_t / delta_x)
dy = np.square(c * delta_t / delta_y)
dz = np.square(c * delta_t / delta_z)


alpha = (1 + 2*dx + 2*dy + 2*dz)


# Coefficient matrix

A_diag = np.diag(np.ones((k,)) * alpha)
A_diag_upper_dx = np.diag(np.ones((k-1, )) * -dx, 1)
A_diag_upper_dz = np.diag(np.ones((k-(nx*ny), )) * -dz, nx*ny)

## For zeros in dx
for i in range(ny*nz):
    A_diag_upper_dx[i*nx-1, i*nx] = 0


## For zeros in dy
temp1 = [1]*((nx)*(ny-1)) 
temp0 = [0]*nx
temp = temp1+temp0

arb = int(np.floor((k - ((nx)*(ny-1)) ) / len(temp)))
vec_dy = temp*arb + temp1

A_diag_upper_dy = np.diag(np.array(vec_dy) * -dy, nx)

A_upper = A_diag_upper_dx + A_diag_upper_dy
A_lower = A_upper.T
A = A_upper + A_lower + A_diag


# Boundary Conditions

U = np.zeros((nx+2, ny+2, nz+2))

U[0, :, :] = 0
U[-1, :, :] = 0
U[:, -1, :] = 0
U[:, 0, :] = 0
U[:, :, 0] = 0
U[:, :, -1] = 0

# Incooperating boundary conditions in b (Ax = b)


bound_vec = np.copy(U)

bound_vec_x = np.zeros(U.shape)

bound_vec_x[1, :, :] = bound_vec[0, :, :]
bound_vec_x[-2, :, :] = bound_vec[-1, :, :]

bound_vec_x = (bound_vec_x[1:-1, 1:-1, 1:-1]).reshape(-1, 1)

bound_vec_y = np.zeros(U.shape)

bound_vec_y[:, 1, :] = bound_vec[:, 0, :]
bound_vec_y[:, -2, :] = bound_vec[:, -1, :]

bound_vec_y = (bound_vec_y[1:-1, 1:-1, 1:-1]).reshape(-1, 1)

bound_vec_z = np.zeros(U.shape)

bound_vec_z[:, :, 1] = bound_vec[:, :, 0]
bound_vec_z[:, :, -2] = bound_vec[:, :, -1]

bound_vec_z = (bound_vec_z[1:-1, 1:-1, 1:-1]).reshape(-1, 1)

# Initial Conditions

x = np.linspace(0, Lx, nx+2).reshape(-1, 1, 1)
y = np.linspace(0, Ly, ny+2).reshape(1, -1, 1)
z = np.linspace(0, Lz, nz+2).reshape(1, 1, -1)

def I(x, y, z):
    return - np.sin(x + y + z) + np.cos(y)

In = I(x, y, z).reshape(nx+2, ny+2, nz+2)

u_0 = np.copy(U)
u_0[1:-1, 1:-1, 1:-1] = In[1:-1, 1:-1, 1:-1]

# Using u_t = 0

u_1 = u_0

# b in Ax = b // Lets assume that boudaries are zero

u_0 = u_0[1:-1, 1:-1, 1:-1].reshape(-1, 1)
u_1 = u_1[1:-1, 1:-1, 1:-1].reshape(-1, 1)


b = 2*u_1 + u_0 + dx*bound_vec_x + dy*bound_vec_y + dz*bound_vec_z


for t in range(nt):
    u = np.linalg.solve(A, b)

    u_0, u_1 = u_1, u
    b = 2*u_1 + u_0 + dx*bound_vec_x + dy*bound_vec_y + dz*bound_vec_z

    U[1:-1, 1:-1, 1:-1] = u.reshape(nx, ny, nz)   
    X, Y, Z = np.meshgrid(x, y, z)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    img = ax.scatter3D(X, Y, Z, c=U.T, alpha=0.7, marker='.')
    cb = fig.colorbar(img)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()