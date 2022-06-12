# import libraries
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
#from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model


def cnn_bt(pretrained_weights=None, input_shape=(256, 256, 1)):
    """
    :param pretrained_weights: if the user have pretained weights of a model use this param
    :param input_shape: input shape of the images
    :return: a cnn model that can be used to identify brain tumours
    """
    model = Sequential()
    model.add(Conv2D(32, 4, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, 4, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout((0.2)))
    model.add(Conv2D(64, 4, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


"""
model = cnn_bt()
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
"""
