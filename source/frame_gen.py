# %%
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil

# %%
#time is in the form XXH:XXm:XXs, so we need to convert it to seconds
def convert_time(time):
    hours = int(time[0:2])
    mins = int(time[4:6])
    secs = int(time[8:10])
    return hours*3600 + mins*60 + secs

# %%
phases = pd.read_excel('../data/phases.xlsx', engine='openpyxl')
print(phases.shape)

# %%
#sort vid_id in reverse alphabetical order, in order to include yale videos in training set
phases = phases.sort_values(by=['vid_id'], ascending=False)
phases.head()

# %%
phases.head()

# %%
#for vid_4, keep only labeler 'kc'
print(phases.shape)
non_kc_vid_4 = phases[phases['vid_id'] == 'vid_4']
non_kc_vid_4 = non_kc_vid_4[non_kc_vid_4['labeler'] != 'kc']
phases = phases.drop(non_kc_vid_4.index)
phases.reset_index(drop=True, inplace=True)
print(phases.shape)

# %%
#rename 'sac_reduction' phase to 'hiatal_dissec'
phases.loc[phases['phase'] == 'sac_reduction', 'phase'] = 'hiatal_dissec'

# %%
#get the number of unique videos
len(phases['vid_id'].unique())

# %%
phases['labeler'].value_counts()

# %%
#strip leading and trailing whitespace from the time_start and time_end columns
phases['time_start'] = phases['time_start'].str.strip()
phases['time_end'] = phases['time_end'].str.strip()

# %%
#for each row in the dataframe, make sure that time_start and time_end are integers
for i in range(len(phases)):
    try:
        start1 = convert_time(phases['time_start'][i])
        end1 = convert_time(phases['time_end'][i])
    except:
        print(i)
        print(phases['time_start'][i])
        print(phases['time_end'][i])

# %%
phase_list = phases['phase'].unique()
phase_list

# %%
#check the total amount of time in the dataset
total_time = 0
for i in range(len(phases)):
    start1 = convert_time(phases['time_start'][i])
    end1 = convert_time(phases['time_end'][i])
    total_time += (end1 - start1)

#get the total time in hours, minutes, and seconds
hours = total_time // 3600
mins = (total_time % 3600) // 60
secs = (total_time % 3600) % 60
print(hours, 'hrs,', mins, 'mins,', secs, 'secs')

# %%
print(len(phases))

# %%
phases['path'] = np.NaN
#if vid_id contains 'vid', then path starts with 'unc/raw_ids', elif vid_id contains 'yale', then path starts with 'yale/raw_ids'
for i, row in phases.iterrows():
    if 'vid' in row['vid_id']:
        phases.loc[i, 'path'] = '../data/unc/raw_ids/' + row['vid_id'] + '.mp4'
    elif 'yale' in row['vid_id']:
        phases.loc[i, 'path'] = '../data/yale/raw_ids/' + row['vid_id'] + '.mp4'
    elif 'rush' in row['vid_id']:
        phases.loc[i, 'path'] = '../data/rush/raw_ids/' + row['vid_id'] + '.mp4'
    else:
        print('error')

# %%
#for each video, check if vid_{}_v2.mp4 exists, if it does, then replace the path with that
for i, row in phases.iterrows():
    if os.path.exists('../data/unc/raw_ids/' + row['vid_id'] + '_v2.mp4'):
        phases.loc[i, 'path'] = '../data/unc/raw_ids/' + row['vid_id'] + '_v2.mp4'
    elif os.path.exists('../data/yale/raw_ids/' + row['vid_id'] + '_v2.mp4'):
        phases.loc[i, 'path'] = '../data/yale/raw_ids/' + row['vid_id'] + '_v2.mp4'
    elif os.path.exists('../data/rush/raw_ids/' + row['vid_id'] + '_v2.mp4'):
        phases.loc[i, 'path'] = '../data/rush/raw_ids/' + row['vid_id'] + '_v2.mp4'
    else:
        pass
phases.head()

# %%
#find values in 'path' that are not strings
phases[phases['path'].apply(lambda x: type(x) != str)]

# %%
#split the number of phases into training, validation, and test sets
train_range = int(len(phases) * 0.8)
val_range = int(len(phases) * 0.9)
test_range = len(phases)
print(train_range, val_range, test_range)

# %%
#delete any existing frames
if os.path.exists('frames'):
    shutil.rmtree('frames')

for i in range(len(phases)):
    if i < train_range:
        vid_id = phases['vid_id'][i]
        vid_fname = phases['path'][i]
        phase = phases['phase'][i]
        time_start = phases['time_start'][i]
        time_end = phases['time_end'][i]
        time_start_sec = convert_time(time_start)
        time_end_sec = convert_time(time_end)
        print(vid_fname)
        #if the phase is 'other', then skip it
        if phase == 'other':
            continue
        #if the phase is not 'oob', then add a 4 second buffer to the start and end times
        elif phase != 'oob':
            time_start_sec += 4
            time_end_sec -= 4
            for i in range(time_start_sec, time_end_sec):
                if i%10 == 0:
                    cap = cv2.VideoCapture(vid_fname)
                    cap.set(cv2.CAP_PROP_POS_MSEC, i*1000)
                    ret, frame = cap.read()
                    if ret:
                        if not os.path.exists('../data/frames/train/{}'.format(phase)):
                            os.makedirs('../data/frames/train/{}'.format(phase))
                        cv2.imwrite('../data/frames/train/{}/{}_{}.jpg'.format(phase, vid_id, i), frame)
                else:
                    continue
                cap.release()
        #if the phase is 'oob', then don't add a buffer
        elif phase == 'oob':
            for i in range(time_start_sec, time_end_sec):
                if i%10 == 0:
                    cap = cv2.VideoCapture(vid_fname)
                    cap.set(cv2.CAP_PROP_POS_MSEC, i*1000)
                    ret, frame = cap.read()
                    if ret:
                        if not os.path.exists('../data/frames/train/{}'.format(phase)):
                            os.makedirs('../data/frames/train/{}'.format(phase))
                        cv2.imwrite('../data/frames/train/{}/{}_{}.jpg'.format(phase, vid_id, i), frame)
                else:
                    continue
                cap.release()
    #validation set
    elif i >= train_range and i < val_range:
        vid_id = phases['vid_id'][i]
        vid_fname = phases['path'][i]
        phase = phases['phase'][i]
        time_start = phases['time_start'][i]
        time_end = phases['time_end'][i]
        time_start_sec = convert_time(time_start)
        time_end_sec = convert_time(time_end)
        print(vid_fname)
        #if the phase is 'other', then skip it
        if phase == 'other':
            continue
        #if the phase is not 'oob', then add a 4 second buffer to the start and end times
        elif phase != 'oob':
            time_start_sec += 4
            time_end_sec -= 4
            for i in range(time_start_sec, time_end_sec):
                if i%10 == 0:
                    cap = cv2.VideoCapture(vid_fname)
                    cap.set(cv2.CAP_PROP_POS_MSEC, i*1000)
                    ret, frame = cap.read()
                    if ret:
                        if not os.path.exists('../data/frames/val/{}'.format(phase)):
                            os.makedirs('../data/frames/val/{}'.format(phase))
                        cv2.imwrite('../data/frames/val/{}/{}_{}.jpg'.format(phase, vid_id, i), frame)
                else:
                    continue
                cap.release()
        #if the phase is 'oob', then don't add a buffer
        elif phase == 'oob':
            for i in range(time_start_sec, time_end_sec):
                if i%10 == 0:
                    cap = cv2.VideoCapture(vid_fname)
                    cap.set(cv2.CAP_PROP_POS_MSEC, i*1000)
                    ret, frame = cap.read()
                    if ret:
                        if not os.path.exists('../data/frames/val/{}'.format(phase)):
                            os.makedirs('../data/frames/val/{}'.format(phase))
                        cv2.imwrite('../data/frames/val/{}/{}_{}.jpg'.format(phase, vid_id, i), frame)
                else:
                    continue
                cap.release()
    elif i >= val_range:
        vid_id = phases['vid_id'][i]
        vid_fname = phases['path'][i]
        phase = phases['phase'][i]
        time_start = phases['time_start'][i]
        time_end = phases['time_end'][i]
        time_start_sec = convert_time(time_start)
        time_end_sec = convert_time(time_end)
        print(vid_fname)
        #if the phase is 'other', then skip it
        if phase == 'other':
            continue
        #if the phase is not 'oob', then add a 4 second buffer to the start and end times
        elif phase != 'oob':
            time_start_sec += 4
            time_end_sec -= 4
            for i in range(time_start_sec, time_end_sec):
                if i%10 == 0:
                    cap = cv2.VideoCapture(vid_fname)
                    cap.set(cv2.CAP_PROP_POS_MSEC, i*1000)
                    ret, frame = cap.read()
                    if ret:
                        if not os.path.exists('../data/frames/test/{}'.format(phase)):
                            os.makedirs('../data/frames/test/{}'.format(phase))
                        cv2.imwrite('../data/frames/test/{}/{}_{}.jpg'.format(phase, vid_id, i), frame)
                else:
                    continue
                cap.release()
        #if the phase is 'oob', then don't add a buffer
        elif phase == 'oob':
            for i in range(time_start_sec, time_end_sec):
                if i%10 == 0:
                    cap = cv2.VideoCapture(vid_fname)
                    cap.set(cv2.CAP_PROP_POS_MSEC, i*1000)
                    ret, frame = cap.read()
                    if ret:
                        if not os.path.exists('../data/frames/test/{}'.format(phase)):
                            os.makedirs('../data/frames/test/{}'.format(phase))
                        cv2.imwrite('../data/frames/test/{}/{}_{}.jpg'.format(phase, vid_id, i), frame)
                else:
                    continue
                cap.release()

# %%
#count how many frames have been generated for each phase and train/test set
count_df = pd.DataFrame(columns=['phase', 'train_test', 'count'])
index_count = 0
for phase in os.listdir('../data/frames/train'):
    # count_df = count_df.append({'phase': phase, 'train_test': 'train', 'count': len(os.listdir('frames/train/{}'.format(phase)))}, ignore_index=True)
    count_df = pd.concat([count_df, pd.DataFrame({'phase': phase, 'train_test': 'train', 'count': len(os.listdir('frames/train/{}'.format(phase)))}, index=[index_count])], ignore_index=False)
    index_count += 1
for phase in os.listdir('../data/frames/test'):
    # count_df = count_df.append({'phase': phase, 'train_test': 'test', 'count': len(os.listdir('frames/test/{}'.format(phase)))}, ignore_index=True)
    count_df = pd.concat([count_df, pd.DataFrame({'phase': phase, 'train_test': 'test', 'count': len(os.listdir('frames/test/{}'.format(phase)))}, index=[index_count])], ignore_index=False)
    index_count += 1
for phase in os.listdir('../data/frames/val'):
    count_df = pd.concat([count_df, pd.DataFrame({'phase': phase, 'train_test': 'val', 'count': len(os.listdir('frames/val/{}'.format(phase)))}, index=[index_count])], ignore_index=False)
    index_count += 1


# %%
count_df

# %%



