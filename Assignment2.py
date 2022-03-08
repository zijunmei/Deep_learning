
import datetime
import errno
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random


import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout, GaussianNoise



config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)

)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def plot_img(data_in):
  """
  Generic image plotting function
  """
  n = 16
  now = datetime.datetime.now()
  plt.figure(figsize=(20,4))
  for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(data_in[i].reshape(32,32,1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.savefig('output/{}.png'.format(now.strftime("%Y%m%d-%H%M")))

def plot_images(x_train):
  for i in range(16):
    print(i)
    plt.subplot(4,4,1+i)
    plt.axis('off')
    plt.imshow(x_train[i, :, :, 0], cmap='gray')
  plt.savefig('asdf.png')
  plt.close()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


print('x_train.shape = ', x_train.shape, ' y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape, ' y_test.shape = ', y_test.shape)

print('data type: ', type(x_train[0][0][0]))
print('label type: ', type(y_train[0][0]))

plot_images(x_train)


def LeNet_impl(noise_layer = 0, noise_level = 1):
  """
  LeNet 5 implementation. Noise level is inputted as an integer to make saving the models easier.
  """
  data_in = keras.Input(shape=(28,28,1))
  kernel_size=(5,5)
  noise_level = noise_level / 100

  x = Conv2D(6, kernel_size, activation='relu', strides = (1,1), input_shape= (28,28,1), padding='same')(data_in)
  if noise_layer == 1:
    x = GaussianNoise(noise_level)(x)
  x = AveragePooling2D(pool_size=(2,2), strides = (2,2))(x)
  if noise_layer == 2:
    x = GaussianNoise(noise_level)(x)
  x = Conv2D(16, kernel_size, activation='relu', strides = (1,1))(x)
  if noise_layer == 3:
    x = GaussianNoise(noise_level)(x)
  x = AveragePooling2D(pool_size=(2,2), strides = (2,2))(x)
  if noise_layer == 4:
    x = GaussianNoise(noise_level)(x)
  x = Conv2D(120, kernel_size, activation='relu', strides = (1,1))(x)
  if noise_layer == 5:
    x = GaussianNoise(noise_level)(x)
  x = Flatten()(x)
  if noise_layer == 6:
    x = GaussianNoise(noise_level)(x)
  x = Dense(120, activation='relu')(x)
  if noise_layer == 7:
    x = GaussianNoise(noise_level)(x)
  x = Dense(84, activation='relu')(x)
  if noise_layer == 8:
    x = GaussianNoise(noise_level)(x)
  x = Dropout(0.5)(x)
  if noise_layer >= 9:
    x = GaussianNoise(noise_level)(x)
  x = Dense(10, activation='softmax')(x)
  lenet_model = keras.Model(inputs = data_in, outputs = x, name = "lenet_model")
  lenet_model.compile(optimizer = 'SGD', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
  lenet_model.summary()
  return lenet_model

def plot_results(fitness):
    plt.plot(fitness.history['accuracy'])
    plt.plot(fitness.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png', bbox_inches='tight')
    plt.show()

    plt.plot(fitness.history['loss'])
    plt.plot(fitness.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower left')
    plt.savefig('loss.png',bbox_inches='tight')
    plt.show()


model_default = LeNet_impl()
default_fitness = model_default.fit(x=x_train, y=y_train, steps_per_epoch=128, epochs=500, verbose=1, validation_data=(x_test, y_test))
model_default.save('./file')
with open('./file/result.txt', 'w') as default_history_file:
    default_history_file.write(json.dumps(default_fitness.history))


for eval_noise_level in range (0, 201, 25):
    for i in range(1,10):
        eval_model = LeNet_impl(noise_layer=i, noise_level=eval_noise_level)
        eval_fit = eval_model.fit(x=x_train, y=y_train, steps_per_epoch=128, epochs=50, verbose=1, validation_data=(x_test, y_test))
        model_filename = './models/noise-{}/layer-{}'.format(eval_noise_level, i)

        if not os.path.exists(os.path.dirname(model_filename)):
            try:
                os.makedirs(os.path.dirname(model_filename))
            except OSError as exc: 
                if exc.errno != errno.EEXIST:
                    raise
        
        eval_model.save('./models/noise-{}/layer-{}'.format(eval_noise_level, i))

        hist_filename = './file/noise-{}/layer-{}.txt'.format(eval_noise_level, i)

        if not os.path.exists(os.path.dirname(hist_filename)):
            try:
                os.makedirs(os.path.dirname(hist_filename))
            except OSError as exc: 
                if exc.errno != errno.EEXIST:
                    raise

        with open(hist_filename, 'w') as hist_file: 
            hist_file.write(json.dumps(eval_fit.history))


def LeNet_impl_fixed(train_layer = 1, show_summary = False):
  """
  LeNet 5 implementation. Noise level is inputted as an integer to make saving the models easier.
  """
  data_in = keras.Input(shape=(28,28,1))
  kernel_size=(5,5)

  x = Conv2D(6, kernel_size, activation='relu', strides = (1,1), input_shape= (28,28,1), padding='same', trainable=(train_layer == 1))(data_in)
  x = AveragePooling2D(pool_size=(2,2), strides = (2,2), trainable=(train_layer == 2))(x)
  x = Conv2D(16, kernel_size, activation='relu', strides = (1,1), trainable=(train_layer == 3))(x)
  x = AveragePooling2D(pool_size=(2,2), strides = (2,2), trainable=(train_layer == 4))(x)
  x = Conv2D(120, kernel_size, activation='relu', strides = (1,1), trainable=(train_layer == 5))(x)
  x = Flatten(trainable=(train_layer == 6))(x)
  x = Dense(120, activation='relu', trainable=(train_layer == 7))(x)
  x = Dense(84, activation='relu', trainable=(train_layer == 8))(x)
  x = Dropout(0.5, trainable=(train_layer == 9))(x)
  x = Dense(10, activation='softmax', trainable=(train_layer == 10))(x)
  lenet_model = keras.Model(inputs = data_in, outputs = x, name = "lenet_model")
  lenet_model.compile(optimizer = 'SGD', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
  if show_summary:
    lenet_model.summary()
  return lenet_model



for i in range(1,10):
    eval_model = LeNet_impl_fixed(i)
    eval_fit = eval_model.fit(x=x_train, y=y_train, steps_per_epoch=128, epochs=50, verbose=1, validation_data=(x_test, y_test))
    model_filename = './models/trainable/layer-{}'.format(i)

    if not os.path.exists(os.path.dirname(model_filename)):
        try:
            os.makedirs(os.path.dirname(model_filename))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise
    
    eval_model.save(model_filename)

    hist_filename = './file/trainable/layer-{}.txt'.format(i)

    if not os.path.exists(os.path.dirname(hist_filename)):
        try:
            os.makedirs(os.path.dirname(hist_filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(hist_filename, 'w') as hist_file: 
        hist_file.write(json.dumps(eval_fit.history))


with open(pathlib.Path('file/result.txt'), 'r') as default_model_history_file:
  default_history_file_contents = json.loads(default_model_history_file.readlines()[0])
  plt.plot(default_history_file_contents['accuracy'])
  plt.plot(default_history_file_contents['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9'], loc='upper left')
plt.savefig('accuracy.png', bbox_inches='tight')
plt.show()

# noise experiment
noise_accuracies = []
for i in range(0,9):
    index = i + 1
    noise_accuracies.append([])
    for noise_level in  range(0, 201, 25):
        file_loc = pathlib.Path('file/noise-{}/layer-{}.txt'.format(noise_level, index))
        with open(file_loc, 'r') as noise_model_history_file:
            noise_history_file_contents = json.loads(noise_model_history_file.readlines()[0])
            noise_accuracies[i].append(noise_history_file_contents['accuracy'][-1])

noise_magnitudes = [0, 0.25, 0.50, 0.75, 1, 1.25, 1.5, 1.75, 2]

for i in range(len(noise_accuracies)):
    plt.plot(noise_magnitudes, noise_accuracies[i])
plt.title('model accuracy vs noise')
plt.ylabel('accuracy')
plt.xlabel('noise amt')
plt.legend(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9'], loc='upper left')
plt.savefig('accuracy-noise.png', bbox_inches='tight')
plt.show()

# training experiment
for i in range(1,10):
    file_loc = pathlib.Path('file/trainable/layer-{}.txt'.format(i))
    with open(file_loc, 'r') as trainable_model_history_file:
        trainable_history_file_contents = json.loads(trainable_model_history_file.readlines()[0])
        plt.plot(trainable_history_file_contents['accuracy'])

plt.title('model accuracy with single trainable weights')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9'], loc='upper left')
plt.savefig('accuracy-trainable.png', bbox_inches='tight')
plt.show()


for i in range(1,10):
    file_loc = pathlib.Path('file/trainable/layer-{}.txt'.format(i))
    with open(file_loc, 'r') as trainable_model_history_file:
        trainable_history_file_contents = json.loads(trainable_model_history_file.readlines()[0])
        plt.plot(trainable_history_file_contents['loss'])

plt.title('model loss with single trainable weights')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9'], loc='upper left')
plt.savefig('loss-trainable.png', bbox_inches='tight')
plt.show()