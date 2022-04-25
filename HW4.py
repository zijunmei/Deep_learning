
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# necessary for my 3080 GPU to ensure it performs as expected and doesn't weirdly run out of memory

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# get imagenet class names
cat_file = open("imagenet1000_clsidx_to_labels.txt", "r")

contents = cat_file.read()
cat_dict = ast.literal_eval(contents)
cat_file.close()
print(len(cat_dict))

# function to turn category number to label
def find_cat_in_dict(prediction_name):
  for cat_num, cat_name in cat_dict.items():
    if prediction_name in cat_name:
      return cat_num

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(avg_pool)
    avg_pool = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)

    max_pool = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(max_pool)
    max_pool = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(max_pool)

    combined = Add()([avg_pool, max_pool])
    combined = Activation('sigmoid')(combined)

    if K.image_data_format == "channels_first":
        combined = Permute((3,1,2))(combined)

    return multiply([input_feature, combined]) 

def spatial_attention(input_feature, kernel_size=7):
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)

    combined = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters = 1, kernel_size = kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(combined)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def CBAM_impl(input, ratio=8):
    x = channel_attention(input, ratio)
    x = spatial_attention(x)
    return x

# initialize VGG16 model
model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
model.summary()
# prevent original VGG16 layers from being trainable with new data
# i = 0
for layer in model.layers:
  layer.trainable = False
#   print(i, layer.name)
#   i = i + 1

def normalize_img(image, label):
  normalized = tf.cast(tf.image.resize(image, (224,224)), tf.float32) / 255., label
  return normalized

# load imagenette images for processing
builder = tfds.builder('imagenette/160px')
builder.download_and_prepare()
datasets = builder.as_dataset(as_supervised=True)
train_data,test_data = datasets['train'],datasets['validation']

# define and prefetch training data
training_dataset = train_data.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
training_dataset = training_dataset.batch(128)
training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# define and prefetch training data
validation_dataset = test_data.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(128)
validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)




for layer_loc in [3,6,10,14,18]:
  # copy off the original model so we can compare later
  model_copy = model

  for il in range(len(model_copy.layers) - 1):
    if il == 0:
      xl = model_copy.layers[il].output
    else:
      xl = model_copy.layers[il](xl)
    # locations of pooling: 3,6,10,14,18
    # can change location accordingly
    if il == layer_loc:
      xl = CBAM_impl(xl)

  # reduced softmax layer (to reduce number of considered categories to 10)
  xl = Dense(10,activation='softmax')(xl)

  # define new model with CBAM block
  new_model= Model(model_copy.input,xl)
  """
  # compile new model
  new_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )


  # fit new data to train CBAM and softmax layers
  new_model.fit(training_dataset,
                validation_data=validation_dataset,
                batch_size=128,
                epochs=10)
  
  new_model.save('./models/cbam-layer-{}'.format(layer_loc))
  # new_model.summary()
"""
cbam_location = 0
for layer in new_model.layers:
    # print('{} {}'.format(cbam_location, layer.name))
    cbam_location = cbam_location + 1

print(len(new_model.layers))
print(len(model.layers))
# cbam_location = 0
# for layer in model.layers:
#     print('{} {}'.format(cbam_location, layer.name))
#     cbam_location = cbam_location + 1

# the tf-keras-vis library only takes an array,
# so we need to load our one image in as an array
input_title = 'horn2'
img1 = load_img(f'inputs/{input_title}.JPEG', target_size=(224, 224))
images = np.asarray([np.array(img1)])
# convert the image to an array
img = img_to_array(img1)
# expand dimensions so that it represents a single 'sample'
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

def plot_feature_maps(feature_maps, layer_location, layer_name):
    plt.figure(figsize=(16,16))
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    plt.savefig(f'outputs/{input_title}-{layer_location}-CBAM-{layer_name}.png')

orig_model_num_layers = len(model.layers)
# for layer_loc in [3,6,10,14,18]:
for layer_loc in [3,6,10,14,18]:
    loaded_model = keras.models.load_model('./models/cbam-layer-{}'.format(layer_loc))
    layer_num_difference = len(loaded_model.layers) - orig_model_num_layers

    featmap_model = Model(inputs=loaded_model.inputs, outputs = loaded_model.layers[layer_loc].output)
    feature_maps = featmap_model.predict(img)
    plot_feature_maps(feature_maps, 'pre', loaded_model.layers[layer_loc].name)

    featmap_model = Model(inputs=loaded_model.inputs, outputs = loaded_model.layers[layer_loc + layer_num_difference].output)
    feature_maps = featmap_model.predict(img)
    plot_feature_maps(feature_maps, 'post', loaded_model.layers[layer_loc].name)