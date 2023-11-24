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

root_directory = 'dataset/'

patch_size = 256

image_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':
        images = os.listdir(path)
        for i, image_name in enumerate(images):
            if image_name.endswith(".png"):
                image = cv2.imread(path+"/"+image_name, 1)
                SIZE_X = 500
                SIZE_Y = 500
                image = Image.fromarray(image)
                image = image.crop((0, 0, SIZE_X, SIZE_Y))
                image = np.array(image)
                print("Nuevo parche de imagen:", path+"/"+image_name)
                patches_img = patchify(
                    image, (patch_size, patch_size, 3), step=128)
                print("Numero de parches de la imagen:", patches_img.shape)
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, :, :]
                        single_patch_img = scaler.fit_transform(
                            single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        single_patch_img = single_patch_img[0]
                        image_dataset.append(single_patch_img)

###########################################################################

mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':
        masks = os.listdir(path)
        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".png"):
                mask = cv2.imread(path+"/"+mask_name, 1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                SIZE_X = 500
                SIZE_Y = 500
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                mask = np.array(mask)
                print("Now patchifying mask:", path+"/"+mask_name)
                patches_mask = patchify(
                    mask, (patch_size, patch_size, 3), step=128)
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]
                        single_patch_mask = single_patch_mask[0]
                        mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

image_number = random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(
    image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()

###########################################################################

Passable_Area = '#3C1098'.lstrip('#')
Passable_Area = np.array(
    tuple(int(Passable_Area[i:i+2], 16) for i in (0, 2, 4)))  # 60, 16, 152

Built_Area = '#8429F6'.lstrip('#')
Built_Area = np.array(
    tuple(int(Built_Area[i:i+2], 16) for i in (0, 2, 4)))  # 132, 41, 246

Green_Area = '#6EC1E4'.lstrip('#')
Green_Area = np.array(
    tuple(int(Green_Area[i:i+2], 16) for i in (0, 2, 4)))  # 110, 193, 228

Road = '#FEDD3A'.lstrip('#')
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4)))  # 254, 221, 58

Unlabeled = '#9B9B9B'.lstrip('#')
Unlabeled = np.array(
    tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4)))  # 155, 155, 155

label = single_patch_mask


def rgb_to_2D_label(label):

    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == Unlabeled, axis=-1)] = 0
    label_seg[np.all(label == Passable_Area, axis=-1)] = 1
    label_seg[np.all(label == Built_Area, axis=-1)] = 2
    label_seg[np.all(label == Green_Area, axis=-1)] = 3
    label_seg[np.all(label == Road, axis=-1)] = 4

    label_seg = label_seg[:, :, 0]

    return label_seg


labels = []

for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Labels encontrados en el dataset: ", np.unique(labels))

image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:, :, 0])
plt.show()

############################################################################

n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, labels_cat, test_size=0.20, random_state=42)

#######################################

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

metrics = ['accuracy', jacard_coef]


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train,
                     batch_size=16,
                     verbose=1,
                     epochs=100,
                     validation_data=(X_test, y_test),
                     shuffle=False)

model.save('model/modelo_entrenado.hdf5')

############################################################

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

model_resnet_backbone = sm.Unet(
    BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')

model_resnet_backbone.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model_resnet_backbone.summary())

history2 = model_resnet_backbone.fit(X_train_prepr,
                                     y_train,
                                     batch_size=16,
                                     epochs=100,
                                     verbose=1,
                                     validation_data=(X_test_prepr, y_test))

###########################################################

history = history1
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

##################################

model = load_model("model/modelo_entrenado.hdf5",
                   custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                   'jacard_coef': jacard_coef})

y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#######################################################################
