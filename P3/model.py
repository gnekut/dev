import csv
import cv2
import numpy as np
from sklearn import model_selection, utils
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D, Cropping2D, Dropout

# ----- GLOBAL PARAMETERS -----
DATA_FILE_PATH = "C:/dev/udacity/carnd/p3/data/driving_log.csv"
DATA_REDUX = 4 # Data reduction factor for subsampliing (1/N)
EPOCHS = 5 # Number of epochs to run the model  for
BATCH = 64 # Batch size
ZEROS_PCT = 0.5 # Percantage of low-/zero-angle turns/images to keep in training/validcation data.


# ----- METHODS -----
def get_images(meta, theta, batch_size, spe):
    '''
    Function: Data generator for fit_generator model. Retrieves images in batches to reduce memory load. Augments data samples by  flipping the image L to R and negating the steering angle. More augmentation (image rotation, translation, etc.) could help further generalize model.
    Inputs: [meta: list of image file paths][theta: list of steering angles, corrected for camera location/perspective][batch_size: number of samples to retrieve per call to the generator][spe: total samples per epoch]
    Output: Array of original and horizontally mirrored images and their corresponding steering angles
    '''
    while 1:
        meta_shuff, theta_shuff = utils.shuffle(meta, theta)
        for shift in range(0, spe, batch_size):
            meta_sample = meta_shuff[shift:shift+batch_size]
            theta_sample = theta_shuff[shift:shift+batch_size]
            imgs, thetas = [], []
            for img, t in zip(meta_sample, theta_sample):
                raw_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                imgs.append(raw_img)
                thetas.append(t)
                # ----- Augment - Flip -----
                imgs.append(np.fliplr(raw_img))
                thetas.append(-t)
            yield (np.array(imgs), np.array(thetas))


def theta_correction(t_in, delta=0.2):
    '''
    Function: Use "central" steering angle to estimate a perceived steering angle for the left and right cameras. Negative angles indicate a left hand turn and therefore the left camera perceives the turn being less dramatic.
    Inputs: [t_in: cental steering angle array, measured in radians][delta: steering angle perception correction, radians]
    Output: Array of corrected steering angles. Length of t_out is three times t_in due to  left, center, and right camera images.
    '''
    t = np.reshape(np.repeat(t_in,3),(-1,3))
    t_out = t + np.array([0, delta, -delta])
    return np.array(t_out)


# ----- IMPORT META INFORMATION -----
'''
- files_clr_raw, theta_clr_raw: arrays of file paths and steering angles read directly from CSV records (clr stands for center, left, right; the order in which the records appear in the CSV)
- (files,theta)_f(train,valid): Filtered data (removed low angles) and split into training and validation datasets
- (train,valid)_sub: Subsampling of data for model to reduce training time (note that I currently do not have access to a GPU)

'''
files_clr_raw, theta_clr_raw = [], []
with open(DATA_FILE_PATH, 'r') as data_file:
    data = csv.reader(data_file)
    for i, r_data in enumerate(data):
        files_clr_raw.extend([r_data[0:3]])
        theta_clr_raw.extend([float(r_data[3])])
print('Original data lengths:  ', 3*len(files_clr_raw))
print('Matching lengths:   ', len(files_clr_raw)==len(theta_clr_raw))
print('\n\n\n')


# ----- REMOVE PCT NON-STEER DATA -----
i_rmv = np.where(np.absolute(np.array(theta_clr_raw))<0.015)[0]
np.random.shuffle(i_rmv)
files_clr = np.delete(np.array(files_clr_raw), i_rmv[0:int((1-ZEROS_PCT)*len(i_rmv))], axis=0)
theta_clr = np.delete(np.array(theta_clr_raw), i_rmv[0:int((1-ZEROS_PCT)*len(i_rmv))], axis=0)
files_ftrain, files_fvalid, theta_ftrain, theta_fvalid = model_selection.train_test_split(np.ndarray.flatten(files_clr), np.ndarray.flatten(theta_correction(theta_clr)), test_size=0.2)
print('Training Length (F):  ', len(files_ftrain))
print('Validation Length (F):   ', len(files_fvalid))
print(files_ftrain[0:2])
print('\n\n\n')


# ----- SUBSAMPLE DATA -----
train_sub = int(len(files_ftrain)/DATA_REDUX)
valid_sub = int(len(files_fvalid)/DATA_REDUX)
ftrain_spe = train_sub - train_sub%BATCH
fvalid_spe = valid_sub - valid_sub%BATCH
files_ftrain, theta_ftrain = utils.shuffle(files_ftrain, theta_ftrain)
files_fvalid, theta_fvalid = utils.shuffle(files_fvalid, theta_fvalid)
files_ftrain, theta_ftrain = files_ftrain[0:ftrain_spe], theta_ftrain[0:ftrain_spe]
files_fvalid, theta_fvalid = files_fvalid[0:fvalid_spe], theta_fvalid[0:fvalid_spe]
print('Training Length (Sub):    ', len(files_ftrain))
print('Validation Length (Sub):   ', len(files_fvalid))
print(files_ftrain[0:2])
print('\n\n\n')


# ----- INITIALIZE GENERATOR -----
train_gen = get_images(files_ftrain, theta_ftrain, BATCH, ftrain_spe)
valid_gen = get_images(files_fvalid, theta_fvalid, BATCH, fvalid_spe)



# ----- CONV. NETWORK MODEL -----
model = Sequential()
# Normalize and zero mean-center
model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160,320,3)))
# Crop image to focus only on road (remove above horizon and hood of car)
model.add(Cropping2D(cropping=((70, 30), (0, 0))))
# (1)
model.add(Conv2D(24, 5, 2, activation='relu'))
model.add(MaxPooling2D())
# (2)
model.add(Conv2D(48, 5, 2, activation='relu'))
model.add(MaxPooling2D())
# (3)
model.add(Conv2D(48, 3, 1, activation='relu'))
model.add(MaxPooling2D())
# (4)
model.add(Conv2D(72, 3, 1, activation='relu'))
# (5)
#model.add(Conv2D(72, 3, 1, activation='relu'))
# (6)
#model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(40))
model.add(Dense(1))


# ----- MODEL RUN -----
# Notes: [1] Number of samples per epoch must be doubled due to image augmentation within generator (fliplr)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit_generator(generator=train_gen, samples_per_epoch=2*ftrain_spe, nb_epoch=EPOCHS, validation_data=valid_gen, nb_val_samples=2*fvalid_spe)


model.save('model.h5')
