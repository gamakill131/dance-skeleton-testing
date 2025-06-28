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
PROXIMITY_THRESHOLD = 0.08 # 8% of image diagonal

# COCO Keypoint indices
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
(ds_train, ds_val), ds_info = tfds.load(
    'coco/2017',
    split=['train', 'validation'],
    with_info=True,
    shuffle_files=True
)
print("COCO dataset loaded.")

def process_and_filter_sample(sample):
    """
    This function performs both validation and preprocessing in one step.
    It returns the processed image, keypoints, and a boolean flag
    indicating if the sample is a valid "hard case" for our fine-tuning.
    """
    is_hard_case = False 
    
    if ('objects' in sample and 
        'keypoints' in sample['objects'] and 
        tf.shape(sample['objects']['keypoints'])[0] == 1):
        
        keypoints = sample['objects']['keypoints'][0]

        left_ankle_occluded = (keypoints[LEFT_ANKLE, 2] == 1)
        right_ankle_occluded = (keypoints[RIGHT_ANKLE, 2] == 1)
        left_knee_occluded = (keypoints[LEFT_KNEE, 2] == 1)
        right_knee_occluded = (keypoints[RIGHT_KNEE, 2] == 1)
        left_wrist_occluded = (keypoints[LEFT_WRIST, 2] == 1)
        right_wrist_occluded = (keypoints[RIGHT_WRIST, 2] == 1)
        
        is_occluded = (left_ankle_occluded or right_ankle_occluded or
                       left_knee_occluded or right_knee_occluded or
                       left_wrist_occluded or right_wrist_occluded)

        la_coords = keypoints[LEFT_ANKLE, :2]
        ra_coords = keypoints[RIGHT_ANKLE, :2]
        lk_coords = keypoints[LEFT_KNEE, :2]
        rk_coords = keypoints[RIGHT_KNEE, :2]
        lw_coords = keypoints[LEFT_WRIST, :2]
        rw_coords = keypoints[RIGHT_WRIST, :2]
        ls_coords = keypoints[LEFT_SHOULDER, :2]
        rs_coords = keypoints[RIGHT_SHOULDER, :2]

        def squared_dist(p1, p2):
            return tf.reduce_sum(tf.square(p1 - p2))

        is_crossed_legs = (squared_dist(la_coords, rk_coords) < PROXIMITY_THRESHOLD**2 or
                           squared_dist(ra_coords, lk_coords) < PROXIMITY_THRESHOLD**2)
        
        is_crossed_arms = (squared_dist(lw_coords, rs_coords) < PROXIMITY_THRESHOLD**2 or
                           squared_dist(rw_coords, ls_coords) < PROXIMITY_THRESHOLD**2)

        if is_occluded or is_crossed_legs or is_crossed_arms:
            is_hard_case = True
            
    # The MoveNet model from TF Hub expects a tf.int32 tensor.
    image = tf.image.resize_with_pad(sample['image'], INPUT_SIZE, INPUT_SIZE)
    image = tf.cast(image, dtype=tf.int32)
    
    if is_hard_case:
        final_keypoints = sample['objects']['keypoints'][0]
    else:
        final_keypoints = tf.zeros((17, 3), dtype=tf.float32)

    return image, final_keypoints, is_hard_case

print("Applying robust filter-map process...")

# 1. Map the combined function to every sample
ds_train_processed = ds_train.map(process_and_filter_sample, num_parallel_calls=tf.data.AUTOTUNE)
ds_val_processed = ds_val.map(process_and_filter_sample, num_parallel_calls=tf.data.AUTOTUNE)

# 2. Filter out the samples that were flagged as not being hard cases
ds_train_hard = ds_train_processed.filter(lambda image, keypoints, is_hard: is_hard)
ds_val_hard = ds_val_processed.filter(lambda image, keypoints, is_hard: is_hard)

# 3. Final map to remove the boolean flag, leaving a (image, keypoints) dataset
train_ds = ds_train_hard.map(lambda image, keypoints, is_hard: (image, keypoints))
val_ds = ds_val_hard.map(lambda image, keypoints, is_hard: (image, keypoints))

# Apply batching and prefetching
train_ds = train_ds.take(TRAIN_SAMPLES_TO_USE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.take(VAL_SAMPLES_TO_USE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Datasets are filtered, preprocessed, and ready for training.")


# --- 3. MODEL BUILDING (NEW, MORE ROBUST APPROACH) ---

class MovenetFinetuner(tf.keras.Model):
    """
    A custom Keras Model for fine-tuning MoveNet.
    This approach is more stable than using hub.KerasLayer directly.
    """
    def __init__(self, model_handle, trainable=True):
        super().__init__()
        # Load the TF Hub model as a callable object
        self.hub_model = hub.load(model_handle)
        
        # We can't set trainable=True directly on the loaded object.
        # Instead, we wrap it in a custom model and the variables
        # will be trainable by default.
        
    def call(self, inputs):
        """
        Defines the forward pass.
        The loaded TF Hub model has a specific signature that needs to be called.
        """
        # The 'serving_default' signature expects a 'keys' argument.
        # It returns a dictionary of outputs.
        model_output = self.hub_model.signatures['serving_default'](tf.cast(inputs, dtype=tf.int32))
        return model_output['output_0']

def custom_loss(y_true, y_pred):
    """
    Custom loss function to handle MoveNet's output.
    """
    y_pred = tf.squeeze(y_pred, axis=[1, 2])
    true_coords = y_true[:, :, :2]
    pred_coords = y_pred[:, :, :2]
    mask = tf.cast(y_true[:, :, 2] > 0, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)
    squared_diff = tf.square(true_coords - pred_coords) * mask
    num_valid_kpts = tf.reduce_sum(mask)
    if num_valid_kpts == 0:
      return 0.0
    loss = tf.reduce_sum(squared_diff) / num_valid_kpts
    return loss

print("Building the fine-tuning model with custom Keras Model class...")
model = MovenetFinetuner(MODEL_HANDLE)

# --- 4. TRAINING ---

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss=custom_loss)

print("\nStarting fine-tuning process...")
print(f"Training on up to {TRAIN_SAMPLES_TO_USE} samples for {EPOCHS} epochs.")

# Build the model by passing a dummy input, which is necessary before summary() or training
dummy_input = tf.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.int32)
model(dummy_input)
model.summary()


history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

print("\nFine-tuning complete.")
# To save the model:
# model.save("movenet_finetuned_crossed_limbs")
