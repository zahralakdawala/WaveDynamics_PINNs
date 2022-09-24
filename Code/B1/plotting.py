import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


def heat_map_plot(sol_array, title_array, t, x, titlefile, ymin = -1, ymax = 1, save = True):
        if len(sol_array) == 1:
            fig = plt.figure(figsize=(14,7))
            gs = GridSpec(len(sol_array), 1)
        else:
            fig = plt.figure(figsize=(7,6))
            gs = GridSpec(len(sol_array), 3)
        
        ymin = min(np.array(sol_array).reshape(-1, ))
        ymax = max(np.array(sol_array).reshape(-1, ))
        
        for i in range(len(sol_array)):
            plt.subplot(gs[i, :])
            plt.title(title_array[i])
            plt.xlabel('t')
            plt.ylabel('x')
            vmin, vmax = ymin, ymax
            plt.pcolormesh(t, x, sol_array[i], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))

            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.set_label('u(t,x)')
            cbar.mappable.set_clim(vmin, vmax)
        plt.tight_layout()
        if save:
            plt.savefig('plots/'+ titlefile + '.png')
        plt.show()
        

