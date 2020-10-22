# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:03:02 2020

@author: ning

This script is to create time frequency arrays for spindle events

For k-complex and sleep stages, we could modify this and do it later when we are familar with the data

"""

import os
import mne
import utils
import itertools

from glob import glob
from tqdm import tqdm

import numpy  as np
import pandas as pd

from matplotlib import pyplot as plt

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
    utils.download_EEG_annotation(sub,day,EEG_dir,annotation_dir,)
    # create folder for time frequency for spindles
    if not os.path.exists(os.path.join(time_frequency_dir,f'sub{sub}day{day}_spindle')):
        os.mkdir(os.path.join(time_frequency_dir,f'sub{sub}day{day}_spindle'))

    # some hyper-hyper-parameters
    time_steps          = 250 # in miliseconds
    window_size         = 1000 # in miliseconds

    # load the EEG raw data, and we don't need any pre-processing, such as ICA
    raw                 = utils.load_EEG_raw(vhdr_file_name =
                             glob(
                                os.path.join(
                                        EEG_dir,
                                        f'suj{sub}_*nap_day{day}.vhdr')
                                )[0]
                            )

    # create time segments for cutting overlapping windows
    df_events           = pd.read_csv(os.path.join(annotation_dir,
                                                   f'suj{sub}_day{day}_annotations.txt'))
    # since we don't want to have too many "normal" data (labeled 0),
    # we cut off the last part of EEG when no particular events
    spindle_events      = df_events[df_events['Annotation'] == 'spindle']
    kcomplex_events     = df_events[df_events['Annotation'] == 'k-complex']
    stage_2_sleep_events= df_events[df_events['Annotation'] == 'Marker:Markoff:2']

    # we only look at the data from when the first 2nd stage sleep started
    if len(stage_2_sleep_events) > 1:
        print('stage 2 sleep annotations are provided')
        tmin            = np.min(stage_2_sleep_events['Onset'].values)
    else:
        tmin            = 0
    # and we stop looking at the data when the last spindle, kcomplex, or 2nd stage stops,
    # whichever one happens the latest
    tmax                = np.max([spindle_events['Onset'].values.max(),
                                    kcomplex_events['Onset'].values.max() + 1,
                                    stage_2_sleep_events['Onset'].values.max() + 1,
                                    ]) * raw.info['sfreq']
    onsets              = np.arange(start   = tmin,
                                    stop    = tmax,
                                    step    = time_steps,
                                    )

    offsets             = onsets + window_size

    windows             = np.vstack([onsets,offsets]).T.astype(int)

    # label spindles
    # if a segement of EEG contains a spindle time stamp, it is labeled "1"
    # so we directly use the Pandas DataFrame in the name of "spindle_events"
    # we created above
    spindle_time_stamps = np.array(spindle_events['Onset'].values * 1000,
                                   dtype = 'int')
    labels              = []
    # let's define all spindle lasted for 1.5 seconds and the annotated time stamp was put on the .25 second location
    intervals = [[item-250,item+1250] for item in spindle_time_stamps]

    # if the segmented window overlap any spindle window, it is defined as a spindle segment
    # but, we want to define the "overlap" better, so I also add a term "tolerate"
    # only if the overlapping is more than some minimum requirement -- tolerate -- we can say it is a spindle
    tol = 20 # in milliseconds
    for window in tqdm(windows):
        if np.sum([utils.getOverlap(window,item) for item in intervals]) > tol:
            labels.append(1)
        else:
            labels.append(0)

    event_id            = {'spindle':1,'no spindle':0}
    events              = np.vstack([onsets,
                                     np.zeros(onsets.shape),
                                     np.array(labels)]).T.astype(int)

    # filter the raw data
    raw_filtered = raw.copy().filter(5,30)
    
    # cut the data in segments
    picks               = mne.pick_types(raw_filtered.info,
                                         eeg    = True,
                                         eog    = False,
                                         misc   = False)
    epochs              = mne.Epochs(raw_filtered,
                                     events     = events,
                                     event_id   = event_id,
                                     tmin       = 0,
                                     tmax       = window_size / raw.info['sfreq'],
                                     baseline   = (0,None),
                                     preload    = True,
                                     picks      = picks,
                                     detrend    = 1,
                                     )
    # downsampling
    epochs              = epochs.resample(64)
    # convert the segmented data to time-freqency format
    freqs               = np.arange(1.,31,2)
    n_cycles            = freqs / 2.
    power               = mne.time_frequency.tfr_morlet(epochs,
                                                        freqs       = freqs,
                                                        n_cycles    = n_cycles,
                                                        return_itc  = False,
                                                        average     = False,
                                                        n_jobs      = 2,
                                                        )
    df_pick = pd.DataFrame(events[:,-1].reshape(-1,1),columns = ['labels'])
    df_pick = df_pick[df_pick['labels'] == 0]
    idx = np.random.choice(list(df_pick.index))
    power_array = power.data[idx]
    
    plt.close('all')
    fig,axes = plt.subplots(figsize = (12*4,5*4),
                            nrows = 5,
                            ncols = 12,
                            )
    for ax,temp,title in zip(axes.flatten(),power_array,epochs.ch_names):
        ax.imshow(temp,
                  origin = 'lower',
                  aspect = .025,
                  cmap = plt.cm.coolwarm,
                  extent = [epochs.times.min(),
                            epochs.times.max(),
                            freqs.min(),
                            freqs.max()])
        ax.set(title = title)
    # save the time frequency data by numpy arrays
    for condition in ['spindle','no_spindle']:
        if not os.path.exists(os.path.join(time_frequency_dir,
                                           f'sub{sub}day{day}',
                                           condition)):
            os.mkdir(os.path.join(time_frequency_dir,
                                  f'sub{sub}day{day}',
                                  condition))
    for array,label,window in tqdm(zip(power.data,labels,windows)):
        if label == 1:
            np.save(os.path.join(
                        time_frequency_dir,
                        f'sub{sub}day{day}',
                        'spindle',
                        f'{window[0]}_{window[1]}.npy'),
                    array)
        else:
            np.save(os.path.join(
                        time_frequency_dir,
                        f'sub{sub}day{day}',
                        'no_spindle',
                        f'{window[0]}_{window[1]}.npy'),
                    array)
    
    
    # label k-complex
    # if a segement of EEG contains a spindle time stamp, it is labeled "1"
    # so we directly use the Pandas DataFrame in the name of "kcomplex_events"
    # we created above
    kcomplex_time_stamps = np.array(kcomplex_events['Onset'].values * 1000,
                                   dtype = 'int')
    labels              = []
    # let's define all spindle lasted for 1.5 seconds and the annotated time stamp was put on the .25 second location
    intervals = [[item-250,item+1250] for item in kcomplex_time_stamps]

    # if the segmented window overlap any spindle window, it is defined as a spindle segment
    # but, we want to define the "overlap" better, so I also add a term "tolerate"
    # only if the overlapping is more than some minimum requirement -- tolerate -- we can say it is a spindle
    tol = 20 # in milliseconds
    for window in tqdm(windows):
        if np.sum([utils.getOverlap(window,item) for item in intervals]) > tol:
            labels.append(1)
        else:
            labels.append(0)

    event_id            = {'kcomplex':1,'no kcomplex':0}
    events              = np.vstack([onsets,
                                     np.zeros(onsets.shape),
                                     np.array(labels)]).T.astype(int)

    # filter the raw data
    raw_filtered = raw.copy().filter(1,10)
    
    # cut the data in segments
    picks               = mne.pick_types(raw_filtered.info,
                                         eeg    = True,
                                         eog    = False,
                                         misc   = False)
    epochs              = mne.Epochs(raw_filtered,
                                     events     = events,
                                     event_id   = event_id,
                                     tmin       = 0,
                                     tmax       = window_size / raw.info['sfreq'],
                                     baseline   = (0,None),
                                     preload    = True,
                                     picks      = picks,
                                     detrend    = 1,
                                     )
    # downsampling
    epochs              = epochs.resample(64)
    # convert the segmented data to time-freqency format
    freqs               = np.arange(1.,31,2)
    n_cycles            = freqs / 2.
    power               = mne.time_frequency.tfr_morlet(epochs,
                                                        freqs       = freqs,
                                                        n_cycles    = n_cycles,
                                                        return_itc  = False,
                                                        average     = False,
                                                        n_jobs      = 2,
                                                        )
    df_pick = pd.DataFrame(events[:,-1].reshape(-1,1),columns = ['labels'])
    df_pick = df_pick[df_pick['labels'] == 0] # change for k-complex or not
    idx = np.random.choice(list(df_pick.index))
    power_array = power.data[idx]
    
    plt.close('all')
    fig,axes = plt.subplots(figsize = (12*4,5*4),
                            nrows = 5,
                            ncols = 12,
                            )
    for ax,temp,title in zip(axes.flatten(),power_array,epochs.ch_names):
        ax.imshow(temp,
                  origin = 'lower',
                  aspect = .025,
                  cmap = plt.cm.coolwarm,
                  extent = [epochs.times.min(),
                            epochs.times.max(),
                            freqs.min(),
                            freqs.max()])
        ax.set(title = title)
    
    # save the time frequency data by numpy arrays
    for condition in ['kcomplex','no_kcomplex']:
        if not os.path.exists(os.path.join(time_frequency_dir,
                                           f'sub{sub}day{day}',
                                           condition)):
            os.mkdir(os.path.join(time_frequency_dir,
                                  f'sub{sub}day{day}',
                                  condition))
    for array,label,window in tqdm(zip(power.data,labels,windows)):
        if label == 1:
            np.save(os.path.join(
                        time_frequency_dir,
                        f'sub{sub}day{day}',
                        'kcomplex',
                        f'{window[0]}_{window[1]}.npy'),
                    array)
        else:
            np.save(os.path.join(
                        time_frequency_dir,
                        f'sub{sub}day{day}',
                        'no_kcomplex',
                        f'{window[0]}_{window[1]}.npy'),
                    array)
