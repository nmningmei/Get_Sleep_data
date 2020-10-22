# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:44:12 2020

@author: ning
"""

import os
import mne
import utils

from glob import glob
from tqdm import tqdm

import numpy   as np
import pandas  as pd
import seaborn as sns

from matplotlib import pyplot as plt

sns.set_style('white')

if __name__ == "__main__":
    # where the EEG raw data will be downloaded to
    EEG_dir             = '../EEG'
    # where the annotation for the EEG data will be downloaded to
    annotation_dir      = '../annotations'
    # where the time frequency data will be saved to
    time_frequency_dir  = '../time_freq'
    for f in [EEG_dir,annotation_dir,time_frequency_dir]:
        if not os.path.exists(f):
            os.mkdir(f)
    # let's just pick one of the data to download
    sub,day             = 29,1
    # download the EEG raw signals and the manual-labeled annotations
    utils.download_EEG_annotation(sub,day,EEG_dir,annotation_dir,)
    # create folder for time frequency for spindles
    if not os.path.exists(os.path.join(time_frequency_dir,f'sub{sub}day{day}_spindle')):
        os.mkdir(os.path.join(time_frequency_dir,f'sub{sub}day{day}_spindle'))
    
    FBT = utils.Filter_based_and_thresholding(moving_window_step = .25, # moving step size
                                              moving_window_size = 1.5, # duration of the window
                                              )
    FBT.get_raw(glob(os.path.join(EEG_dir, f'suj{sub}_*nap_day{day}.vhdr'))[0])
    FBT.get_annotation(os.path.join(annotation_dir,f'suj{sub}_day{day}_annotations.txt'))
    FBT.make_events_windows()#time_steps = 3000,window_size = 3000)
    
    # get spindle events
    events = FBT.event_dictionary['spindle']
    event_time_stamps = np.array(events['Onset'].values * FBT.raw.info['sfreq'],dtype = 'int')
    y_true = FBT.label_segments(event_time_stamps)
    FBT.get_epochs(events = y_true,
                   resample = 100, # downsample the data by 90%
                   )
    FBT.get_powers()
    # these are in Numpy arrays
    features = FBT.psds.data # n_sample x 61 channels x 16 frequency bands, x 100 time points
    labels = FBT.epochs.events[:,-1] # <-- we can event define the event by the proportion of overlapping

    df_pick = pd.DataFrame(labels.reshape(-1,1),columns = ['labels'])
    df_pick = df_pick[df_pick['labels'] == 1]
    idx = np.random.choice(list(df_pick.index))
    power_array = features[idx]
    
    plt.close('all')
    fig,axes = plt.subplots(figsize = (12*4,5*4),
                            nrows = 5,
                            ncols = 12,
                            )
    for ax,temp in zip(axes.flatten(),power_array):
        ax.imshow(temp,
                  origin = 'lower',
                  aspect = .025,
                  cmap = plt.cm.coolwarm,
                  extent = [FBT.epochs.times.min(),
                            FBT.epochs.times.max(),
                            FBT.freq.min(),
                            FBT.freq.max()])
    























