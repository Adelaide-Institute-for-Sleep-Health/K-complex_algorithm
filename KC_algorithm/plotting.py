#Copyright (C) 2020 Bastien Lechat

#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
#published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
#of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.


import numpy as np
import matplotlib.pyplot as plt
from sleepAnalysis.utils import EpochData
from mne.stats.parametric import _parametric_ci

def KC_from_probas(C3,onsets,probas,Fs):
    """
    PLot average K-complexes (at time onsets) for different probability thresholds
    """

    post_peak = 1.5  # [s]
    pre_peak = 1.5  # [s]

    colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#e6ab02']
    thresholds = [0.5,0.6,0.7,0.8,0.9]

    for count,th in enumerate(thresholds):

        indexes_th = np.nonzero(np.bitwise_and(probas>th,probas<1.0))
        kc_onset_ths = onsets[indexes_th]

        d = EpochData(kc_onset_ths, C3, post_peak, pre_peak, Fs) * 10**6
        times = np.arange(0, 3, 3 / len(d[0,:])) - 1.5

        ci_ = _parametric_ci(d)
        av = d.mean(axis=0)

        upper_bound = ci_[0].flatten()
        lower_bound = ci_[1].flatten()
        plt.plot(times, av, color=colors[count], label=str(th))
        plt.fill_between(times, upper_bound, lower_bound,
                         zorder=9, color=colors[count], alpha=.2,
                         clip_on=False)
    plt.legend()
    plt.show()

