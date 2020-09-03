from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib import rcParams

import math

# In Windows the next line should provide the full path to convert.exe
# since convert is a Windows command
#rcParams['animation.convert_path'] = "C:\Program Files\ImageMagick-6.9.3\convert.exe"
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 14


red = "#e41a1c"
blue = "#377eb8"
gray = "#eeeeee"


def update(n):
    ax.cla()
    pts = np.random.uniform(low=0, high=1, size=(2, n))
    
    circ = pts[:, pts[0, :]**2 + pts[1, :]**2 <= 1]
    out_circ = pts[:, pts[0, :]**2 + pts[1, :]**2 > 1]
    pi_approx = 4*circ.shape[1]/n
    circle = mpatches.Wedge((0, 0), 1, 0, 90,  color=gray)
    ax.add_artist(circle)
    plt.plot(circ[0, :], circ[1, :], marker='.', markersize=1,
             linewidth=0, color=red)
    plt.plot(out_circ[0, :], out_circ[1, :], marker='.',markersize=1,
             linewidth=0, color=blue)
    plt.title(r"$n = {}, \pi \approx {:.4f}$".format(n, pi_approx))
    plt.axis("square")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    return circ

def gen(n):
    pts = np.random.uniform(low=0, high=1, size=(2, n))
    results = []
    counter = 0
    for i in range(n):
        if pts[0, i]**2 + pts[1, i]**2 <= 1:
            counter += 1
        results.append((counter/(i+1))*4)
    
    return results

def convergence(res):
    pi = math.pi

    error = [abs(pi-element) for element in res]
    logN = [i+1 for i in range(len(res))]

    halfx = [i**(-0.5) for i in logN]

    fig, axes = plt.subplots()
    axes.plot(logN, error, 'y', label=r'Approximation') 
    axes.plot(logN, halfx, '--', color='k', label=r'$O(N^{-1/2})$') 
    axes.set_ylabel(r'Error in $\pi$ Approximation, $\varepsilon$')
    axes.set_xlabel(r'Repititions, $N$')
    axes.set_yscale('log')
    axes.set_xscale('log')
    plt.style.use('classic')
    plt.legend(loc="best", prop={'size': 12})
    
    axes.tick_params(axis="y", direction='in', which='both')
    axes.tick_params(axis="x", direction='in', which='both')

    plt.tight_layout()
    plt.show()



#getcontext().prec = 50
res = gen(1000)
convergence(res)