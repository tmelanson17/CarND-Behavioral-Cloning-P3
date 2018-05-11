import pandas as pd
import os
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm


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

def augment(images, labels):
    im_label_pairs = zip(images, labels)
    im_label_pairs_proc = flipped(im_label_pairs)
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
                    label = batch_sample[self.label_name]
                    images.append(im)
                    angles.append(label)
                
                images, angles = augment(images, angles)

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



def NvidiaBehavioral(inp, n_classes):
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
    fc2 = Dense(50)(fc1)
    fc3 = Dense(10)(fc2)
    return Dense(n_classes)(fc3)


def preprocessing_layers(inp):
    inp_proc = Lambda(lambda x: x / 255. - 0.5)(inp)
    inp_proc = Cropping2D(cropping=((50,20), (0,0)))(inp_proc)
    return inp_proc

def create_model(input_shape):
    inp = Input(input_shape)
    inp_proc = preprocessing_layers(inp)
    out = NvidiaBehavioral(inp_proc, 1)

    model = Model(inp, out)
    model.compile( 'adam' , 'mse')
    # model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.2, shuffle=True)
    return model

def get_callbacks():
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    chkpt_models = './chkpt'
    if not os.path.exists(chkpt_models):
        os.makedirs(chkpt_models)
    chkpt = ModelCheckpoint(chkpt_models, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    return [tensorboard, chkpt]

if __name__ == '__main__':
    
    ## Input Data

    base_dir = '/workspace/media/Udacity/projects/CarND-Behavioral-Cloning-P3/'
    data_dir = base_dir + 'train-straight+curve+recover/'
    data_csv = os.path.join(data_dir, 'driving_log.csv')

    df = pd.read_csv(data_csv, header='infer')

    ## 

    df_split = create_split(df)

    ## Creating the generators

    datagen = DatasetGenerator(len(df_split), data_dir)
    train_generator = datagen.generator(df_split)
    validation_generator = datagen.generator(df_split, mode='validate')

    ##


    ## Training the model

    # input_shape = X_train.shape[1:]
    input_shape = datagen.getshape(df_split)

    model = create_model(input_shape)


    # TODO: Make the callbacks work again
    model.fit_generator(train_generator, steps_per_epoch = len(df_split)*0.8, validation_data=validation_generator, 
                 validation_steps = len(df_split)*0.2, nb_epoch=4)

    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(os.path.join(model_dir, 'model_behavioral_custom_data.h5'))

    ##
