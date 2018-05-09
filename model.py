import pandas as pd
import os
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm


from keras.layers import Dense, Flatten, Input, Lambda
from keras.models import Model, Sequential


def create_split(df):
    images = df[['center', 'left' ,'right']].melt()
    data = df[['steering']]
    data_repeat = pd.concat([data, data+0.2, data-0.2], ignore_index=True)
    df_split = pd.concat([images, data_repeat], axis=1)
    return df_split


from sklearn.utils import shuffle
from tqdm import tqdm 

# Simple generator for now,
# TODO: upgrade to Keras Sequential
class Dataset:
    
    def __init__(self, datasize, img_name = 'value', label_name='steering', validation_split=0.2):
        self.data_dir = '/workspace/media/Udacity/projects/CarND-Behavioral-Cloning-P3/data'
        self.validation_split = validation_split
        self.n_samples = datasize
        self.n_train = int((1 - self.validation_split) * self.n_samples)
        self.n_val = self.n_samples - self.n_train
        self.img_name = img_name
        self.label_name = label_name
        
    def getshape(self, df):
        filename = df[self.img_name][0].split('/')[-1]
        im = imread(os.path.join(self.data_dir, 'IMG', filename))
        return im.shape
    
    def generator(self, df, batch_size=32, mode='train'):
        num_samples = self.n_samples
        while 1: # Loop forever so the generator never terminates
            df = shuffle(df)
            # Split dataset into ()
            init = 0 if mode == 'train' else self.n_train
            end  = self.n_train if mode == 'train' else self.n_samples
            for offset in range(init, end, batch_size):
                batch_samples = df[offset:offset+batch_size]

                images = []
                angles = []
                for index, batch_sample in batch_samples.iterrows():
                    filename = batch_sample[self.img_name].split('/')[-1]
                    im = imread(os.path.join(self.data_dir, 'IMG', filename))
                    images.append(im)
                    angles.append(batch_sample[self.label_name])

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)
                
from keras.layers import Conv2D, MaxPooling2D

def LeNet(inp, n_classes):
    conv1 = Conv2D(6, 5, 5, activation='relu')(inp)
    conv1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(16, 5, 5, activation='relu')(conv1)
    conv2 = MaxPooling2D()(conv2)
    fc0 = Flatten()(conv2)
    fc1 = Dense(120)(fc0)
    fc2 = Dense(84)(fc1)
    return Dense(n_classes)(fc2)


def create_model(input_shape):
    inp = Input(input_shape)
    inp_proc = Lambda(lambda x: x / 255. - 0.5)(inp)
    out = LeNet(inp_proc, 1)

    model = Model(inp, out)
    model.compile( 'adam' , 'mse')
    # model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.2, shuffle=True)
    return model

if __name__ == '__main__':
    
    ## Input Data

    data_dir = '/workspace/media/Udacity/projects/CarND-Behavioral-Cloning-P3/data'
    data_csv = os.path.join(data_dir, 'driving_log.csv')

    df = pd.read_csv(data_csv, header='infer')

    ## 

    df_split = create_split(df)

    ## Creating the generators

    datagen = Dataset(len(df_split))
    train_generator = datagen.generator(df_split)
    validation_generator = datagen.generator(df_split, mode='validate')

    ##


    ## Training the model

    # input_shape = X_train.shape[1:]
    input_shape = datagen.getshape(df_split)

    model = create_model(input_shape)

    model.fit_generator(train_generator, samples_per_epoch= 
                datagen.n_train, validation_data=validation_generator, 
                nb_val_samples=datagen.n_val, nb_epoch=4)

    model.save('model.h5')

    ##