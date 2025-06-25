# Copyright 2025 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np

# --- 1. CONFIGURATION AND CONSTANTS ---

# Constants for the MoveNet model
MODEL_HANDLE = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256  # Input image size for MoveNet.Thunder

# Constants for the dataset and training
BATCH_SIZE = 8
# For demonstration, we'll only use a small subset of the data.
# Remove `.take()` for a full training run.
TRAIN_SAMPLES_TO_USE = 10000
VAL_SAMPLES_TO_USE = 2000
EPOCHS = 5 # Keep epochs low for a quick demo run

# Constants for our "Hard-Case Mining" filter
# This is a normalized distance threshold. If keypoints are closer than
# this percentage of the image diagonal, we consider them "close".
PROXIMITY_THRESHOLD = 0.08 # 8% of image diagonal

# COCO Keypoint indices for left/right side checks
# Refer to COCO documentation for the full 17 keypoint map
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


# --- 2. DATA LOADING AND PREPARATION ---

print("Loading COCO 2017 dataset from TensorFlow Datasets...")
# NOTE: The first time you run this, it will download the COCO dataset,
# which is very large (~19GB) and can take a significant amount of time.
(ds_train, ds_val), ds_info = tfds.load(
    'coco/2017',
    split=['train', 'validation'],
    with_info=True,
    shuffle_files=True
)
print("COCO dataset loaded.")

def filter_crossed_limbs(sample):
    """
    This is the core filtering logic to find "hard cases."
    A sample is considered a hard case if it meets any of these criteria:
    1. A key limb joint (ankle, knee, wrist) is occluded.
    2. The left ankle is physically close to the right knee.
    3. The right ankle is physically close to the left knee.
    4. The left wrist is physically close to the right shoulder.
    5. The right wrist is physically close to the left shoulder.
    """
    person = sample['objects']
    keypoints = person['keypoints']
    
    # We only work with single-person images for simplicity
    if tf.shape(keypoints)[0] != 1:
        return False
        
    keypoints = keypoints[0] # Get the keypoints for the single person

    # Criterion 1: Check for occluded key limb joints
    # Visibility flag: 0=not labeled, 1=occluded, 2=visible
    left_ankle_occluded = (keypoints[LEFT_ANKLE, 2] == 1)
    right_ankle_occluded = (keypoints[RIGHT_ANKLE, 2] == 1)
    left_knee_occluded = (keypoints[LEFT_KNEE, 2] == 1)
    right_knee_occluded = (keypoints[RIGHT_KNEE, 2] == 1)
    left_wrist_occluded = (keypoints[LEFT_WRIST, 2] == 1)
    right_wrist_occluded = (keypoints[RIGHT_WRIST, 2] == 1)
    
    if left_ankle_occluded or right_ankle_occluded or \
       left_knee_occluded or right_knee_occluded or \
       left_wrist_occluded or right_wrist_occluded:
        return True

    # Get keypoint coordinates (y, x)
    # The coordinates are normalized to the image dimensions.
    la_coords = keypoints[LEFT_ANKLE, :2]
    ra_coords = keypoints[RIGHT_ANKLE, :2]
    lk_coords = keypoints[LEFT_KNEE, :2]
    rk_coords = keypoints[RIGHT_KNEE, :2]
    lw_coords = keypoints[LEFT_WRIST, :2]
    rw_coords = keypoints[RIGHT_WRIST, :2]
    ls_coords = keypoints[LEFT_SHOULDER, :2]
    rs_coords = keypoints[RIGHT_SHOULDER, :2]

    # Helper to calculate squared distance
    def squared_dist(p1, p2):
        return tf.reduce_sum(tf.square(p1 - p2))

    # Criterion 2 & 3: Ankles close to opposite knees
    if squared_dist(la_coords, rk_coords) < PROXIMITY_THRESHOLD**2:
        return True
    if squared_dist(ra_coords, lk_coords) < PROXIMITY_THRESHOLD**2:
        return True
        
    # Criterion 4 & 5: Wrists close to opposite shoulders
    if squared_dist(lw_coords, rs_coords) < PROXIMITY_THRESHOLD**2:
        return True
    if squared_dist(rw_coords, ls_coords) < PROXIMITY_THRESHOLD**2:
        return True

    return False


def preprocess_for_movenet(sample):
    """
    Preprocesses a single image and its keypoints for MoveNet training.
    """
    image = tf.cast(sample['image'], dtype=tf.float32)
    image = tf.image.resize_with_pad(image, INPUT_SIZE, INPUT_SIZE)
    
    # MoveNet expects keypoints as (y, x, score) format, normalized to input size.
    # COCO provides (y, x, visibility). We will use visibility as the score.
    keypoints = sample['objects']['keypoints'][0] # Single person
    
    # We will use the keypoints as the label for our supervised training
    return image, keypoints

# Apply the filter and preprocessing
print("Filtering for 'hard cases' (crossed/occluded limbs)...")
ds_train_hard = ds_train.filter(filter_crossed_limbs)
ds_val_hard = ds_val.filter(filter_crossed_limbs)

# Count the number of hard cases found
train_hard_count = ds_train_hard.reduce(0, lambda x, _: x + 1).numpy()
val_hard_count = ds_val_hard.reduce(0, lambda x, _: x + 1).numpy()

print(f"Found {train_hard_count} training images with crossed/occluded limbs.")
print(f"Found {val_hard_count} validation images with crossed/occluded limbs.")


# Prepare the final datasets for training
train_ds = ds_train_hard.take(TRAIN_SAMPLES_TO_USE).map(preprocess_for_movenet).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val_hard.take(VAL_SAMPLES_TO_USE).map(preprocess_for_movenet).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Datasets are filtered, preprocessed, and ready for training.")


# --- 3. MODEL BUILDING ---

def build_finetune_model(model_handle):
    """
    Builds a Keras model that wraps the pre-trained MoveNet model
    and makes it trainable.
    """
    # Load the pre-trained MoveNet from TF Hub
    movenet_layer = hub.KerasLayer(
        model_handle,
        trainable=True, # This is the key to fine-tuning!
        name="movenet"
    )

    inputs = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="input_image")
    
    # The MoveNet layer returns a tensor of shape [batch, 1, 1, 17, 3]
    # The last dimension contains (y, x, confidence_score)
    outputs = movenet_layer(inputs)
    
    model = tf.keras.Model(inputs, outputs, name="movenet_finetune")
    
    return model

def custom_loss(y_true, y_pred):
    """
    Custom loss function to handle MoveNet's output.
    - y_true: Ground truth keypoints from COCO [batch, 17, 3] (y, x, visibility)
    - y_pred: Predicted keypoints from MoveNet [batch, 1, 1, 17, 3] (y, x, score)
    
    We only calculate loss on keypoints that are visible.
    """
    y_pred = tf.squeeze(y_pred, axis=[1, 2]) # Shape: [batch, 17, 3]

    # Get coordinates and visibility mask
    true_coords = y_true[:, :, :2] # y, x
    pred_coords = y_pred[:, :, :2] # y, x
    
    # A keypoint is valid if visibility is > 0 (i.e., it's visible or occluded)
    mask = tf.cast(y_true[:, :, 2] > 0, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1) # Shape: [batch, 17, 1]
    
    # Calculate Mean Squared Error only on valid keypoints
    squared_diff = tf.square(true_coords - pred_coords) * mask
    
    # Normalize loss by the number of valid keypoints in the batch
    num_valid_kpts = tf.reduce_sum(mask)
    if num_valid_kpts == 0:
      return 0.0 # Avoid division by zero
      
    loss = tf.reduce_sum(squared_diff) / num_valid_kpts
    return loss

print("Building the fine-tuning model...")
model = build_finetune_model(MODEL_HANDLE)
model.summary()

# --- 4. TRAINING ---

# Use a low learning rate for fine-tuning to avoid destroying pre-trained weights
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

model.compile(
    optimizer=optimizer,
    loss=custom_loss
)

print("\nStarting fine-tuning process...")
print(f"Training on {TRAIN_SAMPLES_TO_USE} samples for {EPOCHS} epochs.")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

print("\nFine-tuning complete.")
print("You can now save this model and use it for inference.")
print("It should have improved performance on poses with crossed limbs.")

# To save the model:
# model.save("movenet_finetuned_crossed_limbs")

