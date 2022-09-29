import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy import linspace, zeros, ones, pi, sin, cos, exp, log, diag, trace
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from plotting import *
import sys
sys.path.append('../B1')
from benchmark2 import *



def FDM_1D(xarray, tarray, c, bc_start, bc_end, plotting = True):
    # Problem Setup
    ModelInfo = benchmark()

    x_start = ModelInfo.xstart
    x_end   = ModelInfo.xend
    t_start = ModelInfo.tstart
    t_end   = ModelInfo.tend
    nx      = len(xarray)
    nt      = len(tarray)
    
    delta_t = abs(tarray[1] - t_start)
    delta_x = abs(xarray[1] - x_start)
    
    dx = np.square(c * delta_t / delta_x)
    alpha = (1 + 2*dx)

    # Coefficient matrix

    A_diag = np.diag(np.ones((nx,)) * alpha)
    A_diag_upper_dx = np.diag(np.ones((nx-1, )) * -dx, 1)

    A_upper = A_diag_upper_dx
    A_lower = A_upper.T
    A = A_upper + A_lower + A_diag


    # Boundary Conditions

    U = np.zeros((nx+2, 1))

    U[0] = bc_start
    U[-1] = bc_end

    # Incooperating boundary conditions in b (Ax = b)

    bound_vec_x = np.zeros(U.shape)
    bound_vec_x[1] = U[0]
    bound_vec_x[-2] = U[-1]
    bound_vec_x = (bound_vec_x[1:-1]).reshape(-1, 1)


    # Initial Conditions

    x = np.linspace(x_start, x_end, nx+2).reshape(-1, 1)
    
    I = ModelInfo.u0(x).numpy()
    dt_I = ModelInfo.dt_I(x)

    In = I.reshape(nx+2, 1)

    u_0 = np.copy(U)
    u_0[1:-1] = In[1:-1]

    # Using u_t = 0

    u_1 = u_0 + delta_t*dt_I

    u_0 = u_0[1:-1].reshape(-1, 1)
    u_1 = u_1[1:-1].reshape(-1, 1)
    
    # Plotting
        
    U_sol   = np.zeros((nx+2, nt))
    U_exact = np.zeros((nx+2, nt))
    
    for i, t in enumerate(tarray):
        b = 2*u_1 - u_0 + dx*bound_vec_x 
        u = np.linalg.solve(np.copy(A), np.copy(b))
        
        u_0, u_1         = u_1, u
        U[1:-1]          = u.reshape(nx, 1)
        U_sol[:, i]      = U.reshape(-1,)
        U_exact[1:-1, i] = ModelInfo.u0(xarray, t)
        
    
    if plotting:
        # Simple plot
        simple_plot(x, tarray, sol_array = [U_exact, U_sol], title = 'FDM vs Exact', label_array = ['Exact', 'Numerical'])
        # plot u(t,x) distribution as a color-map
        t_req, x_req = np.meshgrid(tarray, x)
        heat_map_plot([U_sol, U_exact], ['FDM Plot', 'Exact Solution'], t_req, x_req)
       
        
    return U_sol, U_exact
    
  







