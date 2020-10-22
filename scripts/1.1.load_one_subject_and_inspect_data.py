# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:00:49 2020

@author: ning

This script is to show how to load the EEG and annotation data for 
one subject.

You can modify the line with "sub,day = " according to the available
subjects


"""

import os

import pandas as pd
import numpy as np
import seaborn as sns

import mne
import requests 

from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt

sns.set_style('white')
sns.set_context('poster')

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size),
                          desc = f'downloading {save_path.split("/")[-1]} ...'):
            fd.write(chunk)
if __name__ == '__main__':
    # download the data if not
    dataframe_dir   = '../data'
    df              = pd.read_csv(os.path.join(dataframe_dir,'available_subjects.csv'))
    EEG_dir         = '../EEG'
    annotation_dir  = '../annotations'
    for f in [EEG_dir,annotation_dir]:
        if not os.path.exists(f):
            os.mkdir(f)
    
    sub,day = 29,1
    row     = df[np.logical_and(
                df['sub'] == sub,
                df['day'] == day)]
    
    url_eeg         = row['link'].values[0]
    url_vmrk        = row['link'].values[1]
    url_vhdr        = row['link'].values[2]
    url_annotation  = row['annotation_file_link'].values[0]
    
    if len(os.listdir(EEG_dir)) < 1:
        for url in [url_eeg,url_vmrk,url_vhdr]:
            download_url(url,
                         os.path.join(EEG_dir,url.split('/')[-1],)
                         )
    download_url(url_annotation,
                 os.path.join(annotation_dir,
                              f'suj{sub}_day{day}_annotations.txt'))
    
    # load the data
    raw = mne.io.read_raw_brainvision(
            glob(os.path.join(EEG_dir,'*.vhdr'))[0],
            preload = True)
    # set the EOG channels
    channel_types = {'LOc':'eog','ROc':'eog','Aux1':'misc'}
    raw.set_channel_types(channel_types)
    
    raw_ref ,_  = mne.set_eeg_reference(raw,
                                        ref_channels     = 'average',
                                        projection       = True,)
    raw_ref.apply_proj() # it might tell you it already has been re-referenced, but do it anyway
    
    # read standard montage - montage is important for visualization
    montage = mne.channels.read_montage('standard_1020',ch_names=raw.ch_names);#montage.plot()
    raw.set_montage(montage)
    # print some information about the data
    print(raw.info)
    
    # plot a small chunk of the data
    raw.plot(duration = 10,
             start = 50,
             n_channels = 6,
             scalings = dict(eeg=20e-6, eog=150e-6,),
             highpass = 1.,
             lowpass = 30.,
             )
    
    # plot the spindles
    events = pd.read_csv(os.path.join(annotation_dir,
                                      f'suj{sub}_day{day}_annotations.txt'))
    spindle_events = events[events['Annotation'] == 'spindle']
    event_array = np.vstack([spindle_events['Onset'].values,
                             [0] * spindle_events.shape[0],
                             [1] * spindle_events.shape[0]]).T.astype(int)
    
    # filter the raw data
    raw.filter(1,30)
    
    
    # cut the data in segments
    picks = mne.pick_types(raw.info,eeg = True, eog = False, misc = False)
    epochs = mne.Epochs(raw,
                        events = event_array,
                        event_id = {'spindle':1},
                        tmin = -0.5,
                        tmax = 1.5,
                        baseline = (-0.5,-0.2),
                        preload = True,
                        picks = picks,
                        detrend = 1,
                        )
    # downsampling
    epochs = epochs.resample(128)
    # let's see how a spindle look like on average
    evoked = epochs.average()
    evoked.filter(12,14)
    evoked.plot_joint()
    
    # convert the segmented data to time-freqency format
    freqs = np.arange(1.,31,2)
    n_cycles = freqs / 2.
    power = mne.time_frequency.tfr_morlet(epochs,
                                          freqs = freqs,
                                          n_cycles = n_cycles,
                                          return_itc = False,
                                          average = False,
                                          n_jobs = 2,
                                          )
    # visualize one of the spindles (not averaging)
    example = power.data[np.random.randint(1,event_array.shape[0])]
    fig,ax = plt.subplots(figsize = (9,8))
    im = ax.imshow(example.mean(0),
                   origin = 'lower',
                   aspect = .05,
                   cmap = plt.cm.coolwarm,
                   extent = [epochs.times.min(),
                             epochs.times.max(),
                             freqs.min(),
                             freqs.max()]
                   )
    plt.colorbar(im)
    ax.set(xlabel = 'Time (sec)',
           ylabel = 'Frequency (Hz)',
           title = 'Example of spindle averaged of all channels')
    
    # k-complex
    kcomplex_events = events[events['Annotation'] == 'k-complex']
    event_array = np.vstack([kcomplex_events['Onset'].values,
                             [0] * kcomplex_events.shape[0],
                             [1] * kcomplex_events.shape[0]]).T.astype(int)

    # cut the data in segments
    picks = mne.pick_types(raw.info,eeg = True, eog = False, misc = False)
    epochs = mne.Epochs(raw,
                        events = event_array,
                        event_id = {'k-complex':1},
                        tmin = -0.5,
                        tmax = 1.5,
                        baseline = (-0.5,-0.2),
                        preload = True,
                        picks = picks,
                        detrend = 1,
                        )
    epochs = epochs.resample(128)
    # let's see how a spindle look like on average
    evoked = epochs.average()
    evoked.filter(1,5)
    evoked.plot_joint()
    
    # convert the segmented data to time-freqency format
    freqs = np.arange(1.,31,2)
    n_cycles = freqs / 2.
    power = mne.time_frequency.tfr_morlet(epochs,
                                          freqs = freqs,
                                          n_cycles = n_cycles,
                                          return_itc = False,
                                          average = False,
                                          n_jobs = 2,
                                          )
    # visualize one of the spindles (not averaging)
    example = power.data[np.random.randint(1,event_array.shape[0])]
    fig,ax = plt.subplots(figsize = (9,8))
    im = ax.imshow(example.mean(0),
                   origin = 'lower',
                   aspect = .05,
                   cmap = plt.cm.coolwarm,
                   extent = [epochs.times.min(),
                             epochs.times.max(),
                             freqs.min(),
                             freqs.max()]
                   )
    plt.colorbar(im)
    ax.set(xlabel = 'Time (sec)',
           ylabel = 'Frequency (Hz)',
           title = 'Example of k-complex averaged of all channels')














