def shallow_water_1d_test (benchmark, xend, tend):

#
  import matplotlib.pyplot as plt
  import numpy as np

#  Set parameters.
#
  nx = 41
  nt = 100
  x_length = xend
  t_length = tend
  g = 9.8
#
#  Compute H and UH.
#
  h_array, uh_array, x, t = shallow_water_1d ( nx, nt, x_length, t_length, g, benchmark )

  x_min = min ( x )
  x_max = max ( x )
  
  h_min = 0.0
  h_max = np.amax ( h_array )
 
  uh_max = np.amax ( uh_array )
  uh_min = np.amin ( uh_array )
  
  return h_array, uh_array, x, t

def shallow_water_1d ( nx, nt, x_length, t_length, g, benchmark ):

#
  import numpy as np
  import platform

#  Allocate vectors.
#
  h = np.zeros ( nx )
  uh = np.zeros ( nx )
  hm = np.zeros ( nx - 1 )
  uhm = np.zeros ( nx - 1 )
  x = np.zeros ( nx )
  t = np.zeros ( nt + 1 )
  h_array = np.zeros ( [ nx, nt + 1 ] )
  uh_array = np.zeros ( [ nx, nt + 1 ] )
#
#  Define the locations of the nodes and time steps and the spacing.
#
  x = np.linspace ( 0, x_length, nx )
  t = np.linspace ( 0, t_length, nt + 1 )

  dx = x_length / float ( nx - 1 )
  dt = t_length / float ( nt )
#
#  Apply the initial conditions.
#
  h, uh = initial_conditions ( nx, nt, h, uh, x, benchmark )
#
#  Apply the boundary conditions.
#
  h, uh = boundary_conditions ( nx, nt, h, uh, t[0] )
#
#  Store the first time step into H_ARRAY and UH_ARRAY.
#
  h_array[0:nx,0] = h[0:nx]
  uh_array[0:nx,0] = uh[0:nx]
#
#  Take NT more time steps.
#
  for it in range ( 1, nt + 1 ):
#
#  Take a half time step, estimating H and UH at the NX-1 spatial midpoints.
#
    hm[0:nx-1] = ( h[0:nx-1] + h[1:nx] ) / 2.0 \
      - ( dt / 2.0 ) * ( uh[1:nx] - uh[0:nx-1] ) / dx

    uhm[0:nx-1] = ( uh[0:nx-1] + uh[1:nx] ) / 2.0 \
      - ( dt / 2.0 ) * ( \
        uh[1:nx] ** 2    / h[1:nx]   + 0.5 * g * h[1:nx] ** 2 \
      - uh[0:nx-1] ** 2  / h[0:nx-1] - 0.5 * g * h[0:nx-1] ** 2 ) / dx
#
#  Take a full time step, evaluating the derivative at the half time step,
#  to estimate the solution at the NX-2 nodes.
#
    h[1:nx-1] = h[1:nx-1] \
      - dt * ( uhm[1:nx-1] - uhm[0:nx-2] ) / dx

    uh[1:nx-1] = uh[1:nx-1] \
      - dt * ( \
        uhm[1:nx-1] ** 2  / hm[1:nx-1] + 0.5 * g * hm[1:nx-1] ** 2 \
      - uhm[0:nx-2] ** 2  / hm[0:nx-2] - 0.5 * g * hm[0:nx-2] ** 2 ) / dx
#
#  Update the boundary conditions.
#
    h, uh = boundary_conditions ( nx, nt, h, uh, t[it] )
#
#  Copy data into the big arrays.
#
    h_array[0:nx,it] = h[0:nx]
    uh_array[0:nx,it] = uh[0:nx]


  return h_array, uh_array, x, t 

def boundary_conditions ( nx, nt, h, uh, t ):

#
  bc = 2
#
#  Periodic boundary conditions on H and UH.
#
  if ( bc == 1 ):
    h[0] = h[nx-2]
    h[nx-1] = h[1]
    uh[0] = uh[nx-2]
    uh[nx-1] = uh[1]
#
#  Free boundary conditions on H and UH.
#
  elif ( bc == 2 ):
    h[0] = h[1]
    h[nx-1] = h[nx-2]
    uh[0] = uh[1]
    uh[nx-1] = uh[nx-2]
#
#  Reflective boundary conditions on UH, free boundary conditions on H.
#
  elif ( bc == 3 ):
    h[0] = h[1]
    h[nx-1] = h[nx-2]
    uh[0] = - uh[1]
    uh[nx-1] = - uh[nx-2]
    


  return h, uh

import numpy as np

def damBreak(x, h1 = 1, h2 = 0.5):
    val = []
    step1 = 0.45
    step2 = 0.55

        
    for i in x:
        if i < step1:
            val.append(h1)
        elif i > step2:
            val.append(h2)
        else:
            m = (h2-h1)/(step2-step1)
            c = h2 - m*step2
            val.append( m*i + c)
            
    return val

def initial_conditions ( nx, nt, h, uh, x, benchmark):

#
  import numpy as np
  import matplotlib.pyplot as plt
  
  if benchmark == 1:
      h = 0.5*np.sin(np.pi*x)
  elif benchmark == 2:
      h = 2.0 + np.sin ( 2.0 * np.pi * x )
  elif benchmark == 3:
      h = np.array(damBreak(x))
  elif benchmark == 4:
      h = np.array(damBreak(x, h1 = 1, h2 = 0.1))
  elif benchmark == 5:
      h = np.array(damBreak(x, h1 = 1, h2 = 0.02))
      

      
  uh = np.zeros ( nx )
  #plt.plot(x, h)
  #plt.xlabel ( 'x' )
  #plt.ylabel ( 'h(x,0)' )

  return h, uh
  
def timestamp ( ):

  import time

  t = time.time ( )
  #print ( time.ctime ( t ) )

  return None

