from tensorflow.keras.models import load_model
from simple_multi_unet_model import multi_unet_model, jacard_coef
from keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm


from sklearn.preprocessing import MinMaxScaler

###########################################################################

scaler = MinMaxScaler()

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

model = load_model("model/modelo_entrenado.hdf5",
                   custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                   'jacard_coef':jacard_coef})

original_image = cv2.imread('C:\\Users\\laura\\Downloads\\insumos\\images\\recorte_95.png', 1)

new_size = (512, 512)
test_img = cv2.resize(original_image, new_size)


SIZE_X = 512
SIZE_Y = 512
test_img = Image.fromarray(test_img)
test_img = test_img.crop((0, 0, SIZE_X, SIZE_Y))
image = np.array(test_img)

image_resized = cv2.resize(image, (500, 500))

desired_patch_size = 256

num_rows = SIZE_X // desired_patch_size
num_cols = SIZE_Y // desired_patch_size

patches_img_test = patchify(image, (desired_patch_size, desired_patch_size, 3), step=desired_patch_size)

all_predictions = []

for i in range(num_rows):
    row_predictions = [] 
    for j in range(num_cols):
        single_patch_img_test = patches_img_test[i, j, :, :]
        single_patch_img_test = scaler.fit_transform(single_patch_img_test.reshape(-1, single_patch_img_test.shape[-1])).reshape(single_patch_img_test.shape)
        single_patch_img_test = single_patch_img_test[0]
        image_dataset_test = np.expand_dims(single_patch_img_test, axis=0)

        prediction_test = model.predict(image_dataset_test)
        predicted_img_test = np.argmax(prediction_test, axis=3)[0, :, :]
        row_predictions.append(predicted_img_test)

        plt.figure(figsize=(8, 8))
        plt.subplot(121)
        plt.title('Original Patch')
        plt.imshow(single_patch_img_test)
        plt.subplot(122)
        plt.title('Prediction Patch')
        plt.imshow(predicted_img_test)
        plt.show()

    row_prediction = np.concatenate(row_predictions, axis=1)
    all_predictions.append(row_prediction)

full_prediction = np.concatenate(all_predictions, axis=0)
full_prediction_resized = full_prediction[:500, :500]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Original Image')
plt.imshow(image_resized)
plt.subplot(232)
plt.title('Full Prediction Resized')
plt.imshow(full_prediction_resized)
plt.show()