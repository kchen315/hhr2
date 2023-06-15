# %%
from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121, Xception
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os


# %%
MAX_SEQ_LENGTH = 16  #20
NUM_FEATURES = 2048
IMG_SIZE = 512
DENSE_DIM = 512
NUM_HEADS = 8
DROPOUT = 0.5
EPOCHS = 100


# %%
X_dict = {}
y_dict = {}

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

vid_list = phases['vid_id'].unique()
len(vid_list)

# %%
for vid in vid_list:
    X_dict[vid] = np.load('../data/fe_data/X_data_{}.npy'.format(vid))
    y_dict[vid] = np.load('../data/fe_data/y_data_{}.npy'.format(vid))

# %%
from sklearn.model_selection import train_test_split
#split vid_list into train and test
train_vids, test_vids = train_test_split(vid_list, test_size=0.2)
#split train_vids into train and validation
train_vids, val_vids = train_test_split(train_vids, test_size=0.25)
#stack the npy arrays into X_train, X_val, X_test, y_train, y_val, y_test
X_train = np.vstack([X_dict[vid] for vid in train_vids])
X_val = np.vstack([X_dict[vid] for vid in val_vids])
X_test = np.vstack([X_dict[vid] for vid in test_vids])
y_train = np.hstack([y_dict[vid] for vid in train_vids])
y_val = np.hstack([y_dict[vid] for vid in val_vids])
y_test = np.hstack([y_dict[vid] for vid in test_vids])
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

# %%
NUM_CLASSES = len(np.unique(y_train))

# %%
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


# %%
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


# %%
def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = DENSE_DIM
    num_heads = NUM_HEADS
    classes = NUM_CLASSES

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC(), keras.metrics.AUC(curve='PR')]
    )
    return model


def run_experiment():
    filepath = "../source/vit_checkpoint"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10, min_delta=1e-6, restore_best_weights=True
    )

    model = get_compiled_model()
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
    )

    model.load_weights(filepath)
    res_df = pd.DataFrame(columns=['dataset', 'accuracy', 'AUROC', 'AUPRC'])
    res_df.loc[0] = ['train', history.history['categorical_accuracy'][-1], history.history['auc'][-1], history.history['auc_1'][-1]]
    res_df.loc[1] = ['val', history.history['val_categorical_accuracy'][-1], history.history['val_auc'][-1], history.history['val_auc_1'][-1]]
    _, accuracy, test_auroc, test_auprc = model.evaluate(X_test, y_test)
    res_df.loc[2] = ['test', accuracy, test_auroc, test_auprc]
    
    return model, res_df


# %%
trained_model, res_df = run_experiment()
print('model training complete')

import datetime
today = datetime.date.today()
today_str = today.strftime('%m%d%y')

trained_model.save('../results/models/vit_{}.h5'.format(today_str))
res_df.to_csv('../results/vit_res/vit_res_{}.csv'.format(today_str), index=False)

# %%



