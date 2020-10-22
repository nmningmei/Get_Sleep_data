# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:11:12 2020

@author: ning

This is shameless self-promoting script.

The script shows the filter based thresholding method of annotating the 
sleep data.

"""

import os
import mne
import utils

from glob import glob
from tqdm import tqdm

import numpy  as np
import pandas as pd



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
    
    FBT = utils.Filter_based_and_thresholding(moving_window_step = .25,
                                              moving_window_size = 1.5,)
    FBT.get_raw(glob(os.path.join(EEG_dir, f'suj{sub}_*nap_day{day}.vhdr'))[0])
    FBT.get_annotation(os.path.join(annotation_dir,f'suj{sub}_day{day}_annotations.txt'))
    FBT.make_events_windows()#time_steps = 3000,window_size = 3000)
    events = FBT.event_dictionary['spindle']
    event_time_stamps = np.array(events['Onset'].values * FBT.raw.info['sfreq'],dtype = 'int')
    y_true = FBT.label_segments(event_time_stamps)
    
    FBT.find_onset_duration(lower_threshold = .9, higher_threshold = 3.)
    y_pred,y_prob = FBT.label_segments(np.array(FBT.time_find),np.array(FBT.Duration),return_proba = True)
    from sklearn import metrics
    print(metrics.roc_auc_score(y_true[:,-1],y_pred[:,-1]))
    print(metrics.roc_auc_score(y_true[:,-1],y_prob))
    events           = np.vstack([FBT.windows[:,0],
                                  np.zeros(y_true.shape[0]),
                                  y_true[:,-1]
                                  ]).T.astype(int)
    print('stage 2 sleep constraint')
    FBT.sleep_stage_check()
    y_pred,y_prob = FBT.label_segments(np.array(FBT.time_find),np.array(FBT.Duration),return_proba = True)
    print(metrics.roc_auc_score(y_true[:,-1],y_pred[:,-1]))
    print(metrics.roc_auc_score(y_true[:,-1],y_prob))
#    FBT.get_epochs(events = events,)
#    FBT.fit(labels = y_pred[:,-1],resample = 100,n_jobs = 2)
#    y_pred = FBT.auto_proba
#    print(metrics.roc_auc_score(y_true[:,-1],y_pred))


























