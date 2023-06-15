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
MAX_SEQ_LENGTH = 8  #20
NUM_FEATURES = 1024
IMG_SIZE = 128

EPOCHS = 5

# %%
def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    print(preprocessed.shape)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


# feature_extractor = build_feature_extractor()

# %%
X_data = np.load('../data/X_data.npy', allow_pickle=True)
y_data = np.load('../data/y_data.npy', allow_pickle=True)
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(y_data), mask_token=None
)
print(label_processor.get_vocabulary())
print(label_processor(['hiatal_dissec','eso_mob']))
print(X_data.shape)

# %%
def prepare_all_videos(data, labels):
    num_samples = data.shape[0]
    labels = label_processor(labels).numpy()

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )
    # For each video.
    for i in range(num_samples):
        frames = data[i]
        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )
        # Extract features from the frames of the current video.
        for j in range(8):
            temp_frame_features[0, j, :] = feature_extractor.predict(
                frames[None,j]
            )
        frame_features[i,] = temp_frame_features.squeeze()
        print(i)

    return frame_features, labels

# frame_features, labels = prepare_all_videos(X_data, y_data)

# %%
# print(frame_features.shape)
# np.save('frame_features_Densenet.npy',frame_features)
# np.save('labels.npy',labels)

# %%
X = np.load('frame_features_Densenet.npy', allow_pickle=True)
y = np.load('labels.npy',allow_pickle=True)
print(X.shape)
print(y.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

# %%
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)

# %%
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
    dense_dim = 4
    num_heads = 1
    classes = len(label_processor.get_vocabulary())

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def run_experiment():
    filepath = "../source/vit_checkpoint"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model()
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model

# %%
trained_model = run_experiment()
print('model training complete')

import datetime
today = datetime.date.today()
today_str = today.strftime('%m%d%y')

trained_model.save('../results/models/vit_{}.h5'.format(today_str))


# def predict_action(path):
#     class_vocab = label_processor.get_vocabulary()

#     frames = load_video(os.path.join("test", path))
#     frame_features = prepare_single_video(frames)
#     probabilities = trained_model.predict(frame_features)[0]

#     for i in np.argsort(probabilities)[::-1]:
#         print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
#     return frames


# def prepare_single_video(frames):
#     frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

#     # Pad shorter videos.
#     if len(frames) < MAX_SEQ_LENGTH:
#         diff = MAX_SEQ_LENGTH - len(frames)
#         padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
#         frames = np.concatenate(frames, padding)

#     frames = frames[None, ...]

#     # Extract features from the frames of the current video.
#     for i, batch in enumerate(frames):
#         video_length = batch.shape[0]
#         length = min(MAX_SEQ_LENGTH, video_length)
#         for j in range(length):
#             if np.mean(batch[j, :]) > 0.0:
#                 frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
#             else:
#                 frame_features[i, j, :] = 0.0

#     return frame_features

# # This utility is for visualization.
# # Referenced from:
# # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
# def to_gif(images):
#     converted_images = images.astype(np.uint8)
#     imageio.mimsave("animation.gif", converted_images, fps=10)
#     return embed.embed_file("animation.gif")


# test_video = np.random.choice(test_df["video_name"].values.tolist())
# print(f"Test video path: {test_video}")
# test_frames = predict_action(test_video)
# to_gif(test_frames[:MAX_SEQ_LENGTH])



