# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications import Xception
import cv2


# %%
MAX_SEQ_LENGTH = 16  #20
NUM_FEATURES = 2048
IMG_SIZE = 512
EPOCHS = 10


# %%
# Make model
model = keras.models.load_model('../results/models/model_Xception_imagenet_052223.h5')
base_layers = model.get_layer('xception')

#build a feature extractor using the base layers of the model
def build_feature_extractor():
    feature_extractor = base_layers
    preprocess_input = keras.applications.xception.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

#save the feature extractor
# feature_extractor.save(f'../results/models/fe_Xception_{IMG_SIZE}.h5')

# %%
#load the labels
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf

#time is in the form XXH:XXm:XXs, so we need to convert it to seconds
def convert_time(time):
    hours = int(time[0:2])
    mins = int(time[4:6])
    secs = int(time[8:10])
    return hours*3600 + mins*60 + secs

#Preprocess the table
phases = pd.read_excel('../data/phases(2).xlsx', engine='openpyxl')
phases = phases.sort_values(by=['vid_id'], ascending=False)
non_kc_vid_4 = phases[phases['vid_id'] == 'vid_4']
non_kc_vid_4 = non_kc_vid_4[non_kc_vid_4['labeler'] != 'kc']
phases = phases.drop(non_kc_vid_4.index)
phases.reset_index(drop=True, inplace=True)
phases.loc[phases['phase'] == 'sac_reduction', 'phase'] = 'hiatal_dissec'
#strip leading and trailing whitespace from the time_start and time_end columns
phases['time_start'] = phases['time_start'].str.strip()
phases['time_end'] = phases['time_end'].str.strip()
#if vid_id contains 'vid', then path starts with 'unc/raw_ids', elif vid_id contains 'yale', then path starts with 'yale/raw_ids'
for i, row in phases.iterrows():
    if 'vid' in row['vid_id']:
        phases.loc[i, 'path'] = '../data/unc/raw_ids/' + row['vid_id'] + '.mp4'
    elif 'yale' in row['vid_id']:
        phases.loc[i, 'path'] = '../data/yale/raw_ids/' + row['vid_id'] + '.mp4'
    elif 'rush' in row['vid_id']:
        phases.loc[i, 'path'] = '../data/rush/raw_ids/' + row['vid_id'] + '.mp4'
    elif 'UNC' in row['vid_id']:
        phases.loc[i, 'path'] = '../data/unc/raw_ids/' + row['vid_id'] + '.mp4'
    else:
        print(row['vid_id'])
#for each video, check if vid_{}_v2.mp4 exists, if it does, then replace the path with that
for i, row in phases.iterrows():
    if os.path.exists('../data/unc/raw_ids/' + row['vid_id'] + '_v2.mp4'):
        phases.loc[i, 'path'] = '../data/unc/raw_ids/' + row['vid_id'] + '_v2.mp4'
    elif os.path.exists('../data/yale/raw_ids/' + row['vid_id'] + '_v2.mp4'):
        phases.loc[i, 'path'] = '../data/yale/raw_ids/' + row['vid_id'] + '_v2.mp4'
    elif os.path.exists('../data/yale/raw_ids/' + row['vid_id'] + '_robo.mp4'):
        phases.loc[i, 'path'] = '../data/yale/raw_ids/' + row['vid_id'] + '_robo.mp4'
    else:
        pass


# %%
phases = phases[['vid_id', 'path', 'phase', 'time_start', 'time_end']]
phases.head()

# %%
len(phases)

# %%
MAX_SEQ_LENGTH = 16 #20
NUM_FEATURES = 2048
IMG_SIZE = 512
EPOCHS = 10
len_phases = len(phases)

# %%
def get_fe_img(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    fe_img = feature_extractor.predict(np.expand_dims(img, axis=0))[0]
    return fe_img
def get_fe_imgs(vid_fname, time_start_sec, time_end_sec):
    cap = cv2.VideoCapture(vid_fname)
    j = time_start_sec
    fe_imgs = []
    while j < time_end_sec:
        
        cap.set(cv2.CAP_PROP_POS_MSEC, j*1000)
        success, image = cap.read()
        if success:
            fe_img = get_fe_img(image)
            fe_imgs.append(fe_img)
            if len(fe_imgs) == MAX_SEQ_LENGTH:
                break
        else:
            #if we can't read the frame, then we just repeat the last frame
            fe_imgs.append(fe_imgs[-1])
        j += 0.5
    return fe_imgs

# %%
#define a function that will break up a video into clips of length MAX_SEQ_LENGTH, and return a list of time_start_sec and time_end_sec for each clip that can be used by get_fe_imgs
#with MAX_SEQ_LENGTH of 16, and a frame every 0.5 seconds, this will give us 8 seconds of video for each clip
def get_clips(phase, time_start_sec, time_end_sec):
    if phase != 'oob':
        #add a 4 second buffer to the beginning and end of the video
        time_start_sec = time_start_sec - 4
        time_end_sec = time_end_sec + 4
        total_time = time_end_sec - time_start_sec
        num_clips = int(total_time / MAX_SEQ_LENGTH)
        clips = []
        for i in range(num_clips):
            clip_start = time_start_sec + i*MAX_SEQ_LENGTH
            clip_end = clip_start + MAX_SEQ_LENGTH
            clips.append((clip_start, clip_end))
    elif phase == 'oob':
        #add a 1 second buffer to the beginning and end of the video
        time_start_sec = time_start_sec - 1
        time_end_sec = time_end_sec + 1
        total_time = time_end_sec - time_start_sec
        num_clips = int(total_time / MAX_SEQ_LENGTH)
        clips = []
        for i in range(num_clips):
            clip_start = time_start_sec + i*MAX_SEQ_LENGTH
            clip_end = clip_start + MAX_SEQ_LENGTH
            clips.append((clip_start, clip_end))
    return clips

# %%
vid_list = phases['vid_id'].unique()
for vid in vid_list:
    X_data = []
    y_data = []
    vid_phases = phases[phases['vid_id'] == vid]
    len_vid_phases = len(vid_phases)
    for i in range(len_vid_phases):
        vid_id = phases.loc[i, 'vid_id']
        vid_fname = phases.loc[i, 'path']
        phase = phases.loc[i, 'phase']
        time_start_sec = convert_time(phases.loc[i, 'time_start'])
        time_end_sec = convert_time(phases.loc[i, 'time_end'])
        clip_list = get_clips(phase, time_start_sec, time_end_sec)
        for clip in clip_list:
            time_start_sec = clip[0]
            time_end_sec = clip[1]
            fe_imgs = get_fe_imgs(vid_fname, time_start_sec, time_end_sec)
            X_data.append(fe_imgs)
            y_data.append(phase)
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    print(f'{vid} done, captured {len(X_data)} clips')
    
    #save the data
    np.save(f'../data/fe_data/X_data_{vid}.npy', X_data)
    np.save(f'../data/fe_data/y_data_{vid}.npy', y_data)