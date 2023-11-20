import os
import numpy as np
from glob import glob
import nibabel

train_img_dir = "YOUR_TRAIN_IMAGE_FOLDER_PATH"
train_mask_dir = "YOUR_TRAIN_MASK_FOLDER_PATH"
test_img_dir = 'YOUR_TEST_IMAGE_FOLDER_PATH'

image_rows = int(512/2)
image_cols = int(512/2)  

def create_train_data():
    print('-'*30)
    print('Creating training data...')
    print('-'*30)

    train_img = os.path.join(train_img_dir, '*.nii')
    train_label = os.path.join(train_mask_dir, '*.nii')

    images = sorted(glob(train_img))
    labels = sorted(glob(train_label))

    imgs_train = []
    masks_train = []

    for label, img in zip(labels, images):
        training_mask = nibabel.load(os.path.join(train_label, label))
        training_image = nibabel.load(os.path.join(train_img, img))

        for k in range(training_mask.shape[2]-1):
            mask_2d = np.array(training_mask.get_fdata()[::2, ::2, k])
            image_2d = np.array(training_image.get_fdata()[::2, ::2, k])

            if len(np.unique(mask_2d)) != 1:
                masks_train.append(mask_2d) 
                imgs_train.append(image_2d) 

    imgs = np.ndarray(
            (len(imgs_train), image_rows, image_cols), dtype='uint8'
            )
    

    imgs_mask = np.ndarray(
                (len(masks_train), image_rows, image_cols), dtype='uint8'
                )            

    for index, img in enumerate(imgs_train):
        imgs[index, :, :] = img
        
    for index, img in enumerate(masks_train):
        imgs_mask[index, :, :] = img
    
    np.save('imgs_train.npy', imgs)
    np.save('masks_train.npy', imgs_mask)
    print('Saving to .npy files done.')

def create_test_data():
    print('-'*30)
    print('Creating test data...')
    print('-'*30)
    
    images = os.listdir(test_img_dir)   
    imgs_test = []
    
    for image_name in images:
        img = nibabel.load(os.path.join(test_img_dir, image_name))
        image_2d = img.get_fdata()[::2, ::2, :]
        
        for k in range(img.shape[2]):  
            image_slice = np.array(image_2d[:, :, k])
            imgs_test.append(image_slice)
                      
    imgst = np.ndarray(
            (len(imgs_test), image_rows, image_cols), dtype=np.uint8
            )
    
    for index, img in enumerate(imgs_test):
        imgst[index, :, :] = img

    np.save('imgs_test.npy', imgst)
    print('Saving to .npy files done.')

create_train_data()

print('Creating train data was completed')

create_test_data()

print('Creating test data was completed')