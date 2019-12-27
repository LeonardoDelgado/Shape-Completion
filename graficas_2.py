# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:13:36 2019

@author: delga
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

y = np.load('resultados_proposal.npy')

plt.hist(y, bins=10, alpha=1, edgecolor = 'black',  linewidth=1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=y.shape[0]))
plt.gca().set_xlabel('Mean squared error')
plt.grid(True)
plt.rc('legend', fontsize=SMALL_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

plt.savefig('Resultados_proposal.png', dpi = 220)
plt.show()
plt.clf()
np.mean(y)