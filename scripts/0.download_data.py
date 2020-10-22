# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:13:40 2020

@author: ning

This script contains functions and workflows that help downloading 
the EEG raw data and the biomarker annotations. This script does NOT 
download the data but to get the prerequired information for downloading

To download the EEG data with its corresponding annotation file,
please use "wget" + the corresponding links in the df_available
variable.

There is another script in the repository that can download the 
data for you.

"""

import os
import re
import json

import numpy  as np
import pandas as pd

def get_data_info():
    with open('../data/chav7_data.json','r+') as json_file:
        data = json.load(json_file)
        json_file.close()
    
    data        = data['data']
    
    file_names  = [item['id'].split('/')[-1] for item in data[1:] if \
                  ('eeg'  in item['id'].split('/')[-1]) or \
                  ('vhdr' in item['id'].split('/')[-1]) or \
                  ('vmrk' in item['id'].split('/')[-1])]
    file_links  = [item['links']['download'] for item in data[1:] if \
                  ('eeg'  in item['links']['download']) or \
                  ('vhdr' in item['links']['download']) or \
                  ('vmrk' in item['links']['download'])]
    
    df                  = pd.DataFrame(np.vstack([file_names,file_links]).T,columns = ['name','link'])
    temp                = np.vstack(df['name'].map(lambda x:np.array(re.findall(r'\d+',x),dtype = np.int32)).values)
    df['sub']           = temp[:,0]
    df['memory_load']   = temp[:,1]
    df['day']           = temp[:,2]
    
    df                  = df.sort_values(['sub','day','memory_load'])
    
    with open('../data/manual_markers.json','r+') as json_file:
        data = json.load(json_file)
        json_file.close()
    
    data                    = data['data']
    
    file_names              = [item['attributes']['name'] for item in data if ('txt' in item['attributes']['name'])]
    file_links              = [item['links']['download'] for item in data if ('txt' in item['attributes']['name'])]
    df_annotations          = pd.DataFrame(np.vstack([file_names,file_links]).T,
                                           columns = ['name','link'])
    temp                    = np.vstack(df_annotations['name'].map(lambda x:np.array(re.findall(r'\d+',x),dtype = np.int32)).values)
    
    df_annotations['sub']   = temp[:,0]
    df_annotations['day']   = temp[:,1]
    
    temp        = []
    for (sub,day),df_sub in df.groupby(['sub','day']):
        idx_row = np.logical_and(df_annotations['sub'] == sub,
                                 df_annotations['day'] == day,)
        if np.sum(idx_row) == 1:
            row = df_annotations[idx_row]
            
            df_sub['annotation_file_name'] = row['name'].values[0]
            df_sub['annotation_file_link'] = row['link'].values[0]
            
            temp.append(df_sub)
    df_available = pd.concat(temp)
    
    print(f'available datasets = {df_available.shape[0] / 3:.0f}')
    return df_available

if __name__ == '__main__':
    saving_dir  = '../data'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    df          = get_data_info()
    df.to_csv(os.path.join(saving_dir,'available_subjects.csv'),index = False)
