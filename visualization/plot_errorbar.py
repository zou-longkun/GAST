# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

a2w = {'mean': np.array([90.5, 91.7, 92.9, 91.9, 89.9]), 'std': np.array([1.0, 0.6, 0.1, 0.2, 0.3])}
a2d = {'mean': np.array([91.4, 91.6, 93.7, 93.6, 90.1]), 'std': np.array([0.8, 0.2, 0.6, 0.4, 0.2])}
d2a = {'mean': np.array([75.0, 75.1, 77.0, 76.0, 75.8]), 'std': np.array([0.8, 0.4, 0.2, 0.5, 0.7])}
w2a = {'mean': np.array([74.8, 74.3, 76.7, 75.1, 75.1]), 'std': np.array([1.5, 0.2, 0.1, 0.4, 0.6])}

matplotlib.rc('xtick', labelsize=23)
matplotlib.rc('ytick', labelsize=23)

plt.figure(figsize=(8, 8))
ymajorLocator   = MultipleLocator(0.25)
ymajorFormatter = FormatStrFormatter('%1.2f')
yminorLocator   = MultipleLocator(0.05)
plt.rcParams['figure.dpi'] = 300

col=np.array(['1', '2', '4', '8', '16']).astype(str) #

#
plt.errorbar(np.arange(5),a2w['mean'],\
            yerr=a2w['std'],\
            fmt="o:",color="black",ecolor='black',elinewidth=2,capsize=6, ms=6, lw=3, label='A ' + r'$\to$' + ' W')
plt.errorbar(np.arange(5),a2d['mean'],\
            yerr=a2d['std'],\
            fmt="o:",color="red",ecolor='red',elinewidth=2,capsize=6, ms=6, lw=3, label='A ' + r'$\to$' + ' D')
plt.errorbar(np.arange(5),d2a['mean'],\
            yerr=d2a['std'],\
            fmt="o:",color="green",ecolor='green',elinewidth=2,capsize=6, ms=6, lw=3, label='D ' + r'$\to$' + ' A')
plt.errorbar(np.arange(5),w2a['mean'],\
            yerr=w2a['std'],\
            fmt="o:",color="blue",ecolor='blue',elinewidth=2,capsize=6, ms=6, lw=3, label='W ' + r'$\to$' + ' A')
#ax.set_xticklabels(col,rotation=45) #
plt.xticks(np.arange(5), col)
plt.yticks([72, 76, 80, 84, 88, 92, 96])

plt.legend(loc='best', fancybox=True)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=23)

font2 = {'family' : 'Times New Roman',

'weight' : 'normal',

'size'   : 23,

}
plt.xlabel("Value of " + r'$m$', font2)
plt.ylabel("Accuracy (" + r'$\%$' + ")", font2)

plt.show()