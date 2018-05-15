import pandas as pd
import os
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
from time import strftime, localtime

from keras.layers import Dense, Flatten, Input, Lambda, Cropping2D
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint


def create_split(df):
    images = df[['center', 'left' ,'right']].melt()
    data = df[['steering']]
    data_repeat = pd.concat([data, data+0.2, data-0.2], ignore_index=True)
    df_split = pd.concat([images, data_repeat], axis=1)
    return df_split



from sklearn.utils import shuffle
from tqdm import tqdm 
from skimage.io import imread, imshow
import numpy as np

def flipped(im_label_pair_gen):
    im_label_pairs = list(im_label_pair_gen)
    im_label_pairs_flipped = [(np.fliplr(im), -label) for im, label in im_label_pairs]
    return im_label_pairs + im_label_pairs_flipped

def frequency_equalize(im_label_pair_gen):
    im_label_pairs = list(im_label_pair_gen)
    images, labels = zip(*im_label_pairs)
    labels = np.array(labels)
    images = np.array(images)
    hist, bin_edges = np.histogram(labels)
    min_count = np.min(hist)
    equalized_labels = list()
    equalized_images = list()
    for i in range(1, len(bin_edges)):
        indices = np.logical_and( labels < bin_edges[i],
            labels >= bin_edges[i-1] )
        
        equalized_labels.append(labels[indices])
        equalized_images.append(images[indices])
    return zip(np.concatenate(equalized_images), 
        np.concatenate(equalized_labels))
    


def augment(images, labels):
    im_label_pairs = zip(images, labels)
    im_label_pairs_proc = flipped(im_label_pairs)
    # im_label_pairs_proc = frequency_equalize(im_label_pairs_proc)
    return zip(*im_label_pairs_proc)
        

# Simple generator for now,
# TODO: upgrade to Keras Sequential
class DatasetGenerator:
    
    def __init__(self, datasize, data_dir, img_name = 'value', label_name='steering', validation_split=0.2):
        self.data_dir = data_dir
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
            # Split dataset into (train, valid)
            init = self.n_train if mode == 'validate' else 0
            end  = self.n_train if mode == 'train' else self.n_samples
            for offset in range(init, end, batch_size):
                batch_samples = df[offset:offset+batch_size]

                images = []
                angles = []
                for index, batch_sample in batch_samples.iterrows():
                    filename = batch_sample[self.img_name].split('/')[-1]
                    im = imread(os.path.join(self.data_dir, 'IMG', filename))
                    label = batch_sample[self.label_name]
                    images.append(im)
                    angles.append(label)
                
                images, angles = augment(images, angles)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

                
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras import backend as K

def LeNet(inp, n_classes, prob=0.0):
    conv1 = Conv2D(6, 5, 5, activation='relu')(inp)
    conv1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(16, 5, 5, activation='relu')(conv1)
    conv2 = MaxPooling2D()(conv2)
    fc0 = Flatten()(conv2)
    fc1 = Dense(120)(fc0)
    fc2 = Dense(84)(fc1)
    return Dense(n_classes)(fc2)



def NvidiaBehavioral(inp, n_classes, prob=0.0):
    conv1 = Conv2D(24, 5, 5, activation='relu')(inp)
    conv1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(36, 5, 5, activation='relu')(conv1)
    conv2 = MaxPooling2D()(conv2)
    conv3 = Conv2D(48, 5, 5, activation='relu')(conv2)
    conv3 = MaxPooling2D()(conv3)
    conv4 = Conv2D(64, 3, 3, activation='relu')(conv3)
    conv5 = Conv2D(64, 3, 3, activation='relu')(conv4)
    fc0 = Flatten()(conv5)
    fc1 = Dense(100)(fc0)
    fc1 = Dropout(prob)(fc1)
    fc2 = Dense(50)(fc1)
    fc2 = Dropout(prob)(fc2)
    fc3 = Dense(10)(fc2)
    fc3 = Dropout(prob)(fc3)
    fc4 = Dense(n_classes)(fc3)
    return fc4

def preprocessing_layers(inp):
    inp_proc = Lambda(lambda x: x / 255. - 0.5)(inp)
    # Grayscale
    # inp_proc = Lambda(lambda x: K.sum(K.constant([0.21, 0.72, 0.07]) * x, axis=3, keepdims=True))(inp)
    # Cropping variables:
    inp_proc = Cropping2D(cropping=((50,20), (0,0)))(inp_proc)
    return inp_proc

def create_model(input_shape):
    inp = Input(input_shape)
    inp_proc = preprocessing_layers(inp)
    out = NvidiaBehavioral(inp_proc, 1, prob=0.0)

    model = Model(inp, out)
    model.compile( 'adam' , 'mse')
    # model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.2, shuffle=True)
    return model



def get_callbacks():
    
    log_dir = './logs'
    chkpt_dir = './chkpt'
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    timestamp = strftime('%Y-%m-%d-%H:%M:%S', localtime())
    model_name = 'nvidia_behavioral'
    filepath = os.path.join(chkpt_dir, model_name + timestamp)
    
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    chkpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    return [tensorboard, chkpt]

if __name__ == '__main__':
    
    ## Input Data

    base_dir = '/workspace/media/Udacity/projects/CarND-Behavioral-Cloning-P3/'
    train_data_dir = base_dir + 'train-straight+curve+recover/'
    train_data_csv = os.path.join(train_data_dir, 'driving_log.csv')
    # valid_data_dir = base_dir + 'validation/'
    # valid_data_csv = os.path.join(valid_data_dir, 'driving_log.csv')

    df = pd.read_csv(train_data_csv, header='infer')
    # df_valid = pd.read_csv(valid_data_csv, header='infer')

    ## 

    df_split = df
    # df_split = create_split(df)

    ## Creating the generators

    datagen = DatasetGenerator(len(df_split), train_data_dir, img_name='center')
    train_generator = datagen.generator(df_split, mode='train')
    validation_generator = datagen.generator(df_split, mode='validae')

    ##


    ## Training the model

    # input_shape = X_train.shape[1:]
    input_shape = datagen.getshape(df_split)

    model = create_model(input_shape)


    # TODO: Make the callbacks work again
    model.fit_generator(train_generator, steps_per_epoch = 8000, validation_data=validation_generator, 
                        callbacks=get_callbacks(),
                 validation_steps = 2000, nb_epoch=5)

    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(os.path.join(model_dir, 'model-01-control.h5'))

    ##
