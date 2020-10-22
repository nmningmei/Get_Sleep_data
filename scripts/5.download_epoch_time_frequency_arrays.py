# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:35:40 2020

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
    
    info_for_all_subjects_dir   = '../results'
    df                          = pd.read_csv(os.path.join(info_for_all_subjects_dir,'available_subjects.csv'))
    
    target_event                = 'spindle' # change to kcomplex for "k-complex"
    
    for (sub,day),df_sub in df.groupby(['sub','day']):
        
        # download the EEG raw signals and the manual-labeled annotations
        utils.download_EEG_annotation(sub,day,EEG_dir,annotation_dir,)
        # create folder for time frequency
        if not os.path.exists(os.path.join(time_frequency_dir,f'sub{sub}day{day}')):
            os.mkdir(os.path.join(time_frequency_dir,f'sub{sub}day{day}'))
        
        # these 2 lines controls how many examples for training, regardless of the lable 0 or 1
        FBT = utils.Filter_based_and_thresholding(moving_window_step = .5, # moving step size
                                                  moving_window_size = 1.5, # duration of the window
                                                  )
        FBT.get_raw(glob(os.path.join(EEG_dir, f'suj{sub}_*nap_day{day}.vhdr'))[0])
        FBT.get_annotation(os.path.join(annotation_dir,f'suj{sub}_day{day}_annotations.txt'))
        FBT.make_events_windows()#time_steps = 3000,window_size = 3000)
        for target_events in ['spindle','k-complex']:
            # get events
            events = FBT.event_dictionary[target_event] 
            event_time_stamps = np.array(events['Onset'].values * FBT.raw.info['sfreq'],dtype = 'int')
            y_true = FBT.label_segments(event_time_stamps)
            if target_event == 'spindle':
                FBT.find_onset_duration()
                try:
                    FBT.sleep_stage_check()
                except Exception as E:
                    print(E)
                event_time_stamps_pred = FBT.time_find
                y_pred = FBT.label_segments(event_time_stamps_pred)
                for ys,folder_name in zip([y_true,y_pred],['Ground','Predict']):
                    
                    FBT.get_epochs(events = ys,
                                   resample = 100, # when testing the code, I need to downsample, but can be changed to "None" to ignore downsample
                                   )
                    FBT.get_powers()
                    # these are in Numpy arrays
                    powers = FBT.psds.data # n_sample x 61 channels x 16 frequency bands, x 100 time points
                    labels = FBT.epochs.events[:,-1] # <-- we can event define the event by the proportion of overlapping
                    epochs = FBT.epochs.get_data()
                    epochs = epochs.reshape(epochs.shape[0],epochs.shape[1],1,epochs.shape[-1])
                    features = np.concatenate([powers,epochs],2) # concatenate the epochs as an extra "frequency featuer" for power, you can seperate them when fitting the deep neural net
                    windows = FBT.windows
                    
                    # save the time frequency data by numpy arrays
                    for condition in [f'{target_event}_{folder_name}',f'no_{target_event}_{folder_name}']:
                        if not os.path.exists(os.path.join(time_frequency_dir,
                                                           f'sub{sub}day{day}',
                                                           condition)):
                            os.mkdir(os.path.join(time_frequency_dir,
                                                  f'sub{sub}day{day}',
                                                  condition))
                    for array,label,window in tqdm(zip(features,labels,windows)):
                        if label == 1:
                            np.save(os.path.join(
                                        time_frequency_dir,
                                        f'sub{sub}day{day}',
                                        f'{target_event}_{folder_name}',
                                        f'{window[0]}_{window[1]}.npy'),
                                    array)
                        else:
                            np.save(os.path.join(
                                        time_frequency_dir,
                                        f'sub{sub}day{day}',
                                        f'no_{target_event}_{folder_name}',
                                        f'{window[0]}_{window[1]}.npy'),
                                    array)
            elif target_event == 'kcomplex': # kcomplex does not have an algorithm yet
                FBT.get_epochs(events = y_true,
                               resample = 100, # when testing the code, I need to downsample, but can be changed to "None" to ignore downsample
                               )
                FBT.get_powers()
                # these are in Numpy arrays
                powers = FBT.psds.data # n_sample x 61 channels x 16 frequency bands, x 100 time points
                labels = FBT.epochs.events[:,-1] # <-- we can event define the event by the proportion of overlapping
                epochs = FBT.epochs.get_data()
                epochs = epochs.reshape(epochs.shape[0],epochs.shape[1],1,epochs.shape[-1])
                features = np.concatenate([powers,epochs],2) # concatenate the epochs as an extra "frequency featuer" for power, you can seperate them when fitting the deep neural net
                windows = FBT.windows
                # save the time frequency data by numpy arrays
                for condition in [f'{target_event}',f'no_{target_event}']:
                    if not os.path.exists(os.path.join(time_frequency_dir,
                                                       f'sub{sub}day{day}',
                                                       condition)):
                        os.mkdir(os.path.join(time_frequency_dir,
                                              f'sub{sub}day{day}',
                                              condition))
                for array,label,window in tqdm(zip(features,labels,windows)):
                    if label == 1:
                        np.save(os.path.join(
                                    time_frequency_dir,
                                    f'sub{sub}day{day}',
                                    f'{target_event}',
                                    f'{window[0]}_{window[1]}.npy'),
                                array)
                    else:
                        np.save(os.path.join(
                                    time_frequency_dir,
                                    f'sub{sub}day{day}',
                                    f'no_{target_event}',
                                    f'{window[0]}_{window[1]}.npy'),
                                array)



























