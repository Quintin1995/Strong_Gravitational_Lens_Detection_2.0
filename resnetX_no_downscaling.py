from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, GaussianNoise, Conv2D, MaxPooling2D, AveragePooling2D, Add, ZeroPadding2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import models
from tensorflow.keras import backend as K



def build_resnet18_no_downscaling(input_shape, num_outputs, params, optimizer, loss_function, metrics_list):
        
    # Define the input as a Tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((3,3))(X_input)
    
    # Stage 0
    X = Conv2D(64, (7, 7), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(1, 1))(X)

    # Stage 1
    X = _convolutional_block_custom(X, f_size=3, filters = [64, 64, 128], stage=1, block='a', stride=1)
    X = _identity_mapping_block_custom(X, 3, [64, 64, 128], stage=1, block='b')

    # Stage 1
    X = _convolutional_block_custom(X, f_size=3, filters = [128, 128, 256], stage=2, block='a', stride=1)
    X = _identity_mapping_block_custom(X, 3, [128, 128, 256], stage=2, block='b')

    # Stage 1
    X = _convolutional_block_custom(X, f_size=3, filters = [256, 256, 512], stage=3, block='a', stride=1)
    X = _identity_mapping_block_custom(X, 3, [256, 256, 512], stage=3, block='b')

    # AVGPOOL
    if params.use_avg_pooling_2D:
        X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(num_outputs, activation='sigmoid', name='fc_' + str(num_outputs), kernel_initializer = glorot_uniform(seed=0))(X)   # Glorot Uniform should be quite equivalent to He Uniform.
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='resnet50')

    # Compile the Model before returning it.
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)
    print(model.summary(), flush=True)
    return model


# Define a residual block with three convoluational layers. Where the middle convolutional layer has a kernel size of fxf
def _identity_mapping_block_custom(inp, f_size, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Unpack filters. filters is a list of filter amounts per step. (This block contains three steps of the main path, therefore there are three filter amounts.)
    nb_filters1, nb_filters2, nb_filters3 = filters
    
    # Store the input value so that we can add it after the residual block
    inp_shortcut = inp
    
    # Step 1 - main path
    inp = Conv2D(filters = nb_filters1,
                kernel_size=(1,1),
                strides = (1,1),
                padding = 'same',
                name = conv_name_base + "2a",
                kernel_initializer=glorot_uniform(seed=0))(inp)
    inp = BatchNormalization(axis=3, name = bn_name_base + "2a")(inp)   # Axis three is color channel
    inp = Activation('relu')(inp)
    
    # Step 2 - main path - axis refers to the color channel
    inp = Conv2D(filters = nb_filters2,
                kernel_size = (f_size,f_size),
                strides = (1,1),
                padding = 'same',
                name = conv_name_base + '2b',
                kernel_initializer=glorot_uniform(seed=0))(inp)
    inp = BatchNormalization(axis = 3, name = bn_name_base + '2b')(inp)   # Axis three is color channel
    inp = Activation('relu')(inp)
    
    # Step 3 - main path
    inp = Conv2D(filters = nb_filters3,
                kernel_size = (1,1),
                strides = (1,1),
                padding = 'same',
                name = conv_name_base + '2c',
                kernel_initializer = glorot_uniform(seed=0))(inp)
    inp = BatchNormalization(axis = 3, name = bn_name_base + '2c')(inp)   # Axis three is color channel
    
    # Step 4 - Add the shortcut to the main path before the relu activation function
    inp = Add()([inp, inp_shortcut])
    inp = Activation('relu')(inp)
    
    return inp


# This convolutional block is a residual block but can be used with strides aswell. In order to perform downsampling.
def _convolutional_block_custom(inp, f_size, filters, stage, block, stride=1):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Unpack filters. filters is a list of filter amounts per step. (This block contains three steps of the main path, therefore there are three filter amounts.)
    nb_filters1, nb_filters2, nb_filters3 = filters

    # Save the input value
    inp_shortcut = inp

    # Step 1 - main path  - This Step has a stride > 1
    inp = Conv2D(filters=nb_filters1, kernel_size=(1, 1), strides=(stride, stride), padding='same', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(inp)
    inp = BatchNormalization(axis=3, name=bn_name_base + '2a')(inp)   # Axis three is color channel
    inp = Activation('relu')(inp)

    # Step 2 - main path
    inp = Conv2D(filters=nb_filters2, kernel_size=(f_size, f_size), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(inp)
    inp = BatchNormalization(axis=3, name=bn_name_base + '2b')(inp)   # Axis three is color channel
    inp = Activation('relu')(inp)

    # Step 3 - main path
    inp = Conv2D(filters=nb_filters3, kernel_size=(1, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(inp)
    inp = BatchNormalization(axis=3, name=bn_name_base + '2c')(inp)   # Axis three is color channel

    # Step 4 - Shortcut connection   - This Step has a stride > 1
    inp_shortcut = Conv2D(filters=nb_filters3, kernel_size=(1, 1), strides=(stride, stride), padding='same', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(inp_shortcut)
    inp_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(inp_shortcut)   # Axis three is color channel

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    inp = Add()([inp, inp_shortcut])
    inp = Activation('relu')(inp)

    return inp