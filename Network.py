from __future__ import division

import csv
import psutil
import six
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, GaussianNoise, Conv2D, MaxPooling2D, AveragePooling2D, Add, ZeroPadding2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import optimizers, metrics
from tensorflow.keras import backend as K
import tensorflow as tf
import time
from utils import hms, plot_history


class Network:

    def __init__(self, params, datagenerator, training):

        # Set parameters of the Network class
        print("---\nInitializing network #{}...".format(params.net_name), flush=True)
        self.dg                 = datagenerator         # The Network class needs an instance of a datagenerator class.
        self.params             = params                # The Network class needs an instance of the parameter class.
        
        # Set optimizer, Loss function and performancing tracking metric
        self.optimizer          = optimizers.Adam(lr=self.params.net_learning_rate)
        self.loss               = "binary_crossentropy" # Binary crossentropy is one of the best loss functions when it comes to binary classification problems.
        self.metrics            = [metrics.binary_accuracy] if self.params.net_model_metrics == "binary_accuracy" else None
        
        # Define network input/output dimensionality
        self.input_shape        = self.params.img_dims
        self.num_outputs        = self.params.net_num_outputs

        # Define which axis corresponds to which integer, regarding numpy
        self.row_axis           = 1
        self.col_axis           = 2
        self.channel_axis       = 3

        # Case over possible model types
        self.model              = None
        if self.params.net_name == "resnet18":
            self.model          = self.build_resnet(self.input_shape, self.num_outputs, self.basic_block, [2, 2, 2, 2])
        if self.params.net_name == "resnet50":
            self.model          = self.build_resnet50(input_shape = self.input_shape, num_outputs = self.num_outputs)

        # Parameters used when training the model
        self.loss     = []              # Store loss of the model
        self.acc      = []              # Store binary accuracy of the model
        self.val_loss = []              # Store validation loss of the model
        self.val_acc  = []              # Store validation binary accuracy

        # Check if a model is set.
        assert self.model != None

        # A printout of the model to a txt file
        if training:
            self.save_model_to_file()


    # Training function, that will train according to paramters set by the Parameter class
    # A csv writer is opened and being written into during training.
    def train(self):

        # Open a csv writer and write headers into it.
        bufsize = 1
        f = open(self.params.full_path_of_history, "w", bufsize)
        writer = csv.writer(f)
        writer.writerow(["chunk", "loss", "binary_accuracy", "val_loss", "val_binary_accuracy", "time", "cpu_percentage", "ram_usage", "available_mem"])

        # Train the model
        begin_train_session = time.time()       # Records beginning of training time
        try:
            for chunk_idx in range(self.params.num_chunks):
                
                print("chunk {}/{}".format(chunk_idx+1, self.params.num_chunks), flush=True)

                # Load train chunk and targets
                X_train_chunk, y_train_chunk = self.dg.load_chunk(self.params.chunksize, self.dg.Xlenses_train, self.dg.Xnegatives_train, self.dg.Xsources_train, self.params.data_type, self.params.mock_lens_alpha_scaling)
                # Load validation chunk and targets
                X_validation_chunk, y_validation_chunk = self.dg.load_chunk(self.params.validation_chunksize, self.dg.Xlenses_validation, self.dg.Xnegatives_validation, self.dg.Xsources_validation, self.params.data_type, self.params.mock_lens_alpha_scaling)


                # for i in range(10):
                #     plt.imshow(np.squeeze(X_train_chunk[i*100]), origin='lower', interpolation='none', cmap='gray', vmin=0.0, vmax=1.0)
                #     print("train label #{}, idx #{}".format(y_train_chunk[i*100], i*100))
                #     plt.title("train")
                #     plt.show()

                #     plt.imshow(np.squeeze(X_validation_chunk[i*10]), origin='lower', interpolation='none', cmap='gray', vmin=0.0, vmax=1.0)
                #     print("  val label #{}, idx#{}".format(y_validation_chunk[i*10], i*10))
                #     plt.title("val")
                #     plt.show()


                # Fit model on data with a keras image data generator
                network_fit_time_start = time.time()

                # Define a train generator flow based on the ImageDataGenerator
                train_generator_flowed = self.dg.train_generator.flow(
                    X_train_chunk,
                    y_train_chunk,
                    batch_size=self.params.net_batch_size)

                # Define a validation generator flow based on the ImageDataGenerator
                validation_generator_flowed = self.dg.validation_generator.flow(
                    X_validation_chunk,
                    y_validation_chunk,
                    batch_size=self.params.net_batch_size)

                # Fit both generators (train and validation) on the model.
                history = self.model.fit_generator(
                        train_generator_flowed,
                        steps_per_epoch=len(X_train_chunk) / self.params.net_batch_size,
                        epochs=self.params.net_epochs,
                        validation_data=validation_generator_flowed,
                        validation_steps=len(X_validation_chunk) / self.params.net_batch_size)

                print("Training on chunk took: {}".format(hms(time.time() - network_fit_time_start)), flush=True)

                # Save Model self.params to .h5 file
                if chunk_idx % self.params.chunk_save_interval == 0:
                    self.model.save_weights(self.params.full_path_of_weights)
                    print("Saved model weights to: {}".format(self.params.full_path_of_weights), flush=True)

                # Reset backend of tensorflow so that memory does not leak - Clear keras backend when at 75% usage.
                if(psutil.virtual_memory().percent > 75.0):
                    self.reset_keras_backend()

                # Write results to csv file
                writer.writerow(self.format_info_for_csv(chunk_idx, history, begin_train_session))

                # Store loss and accuracy in list
                self.update_loss_and_acc(history)
                
                # Plot loss and accuracy on interval (also validation loss and accuracy) (to a png file)
                if chunk_idx % self.params.chunk_plot_interval == 0:
                    plot_history(self.acc, self.val_acc, self.loss, self.val_loss, self.params)

        except KeyboardInterrupt:
            self.model.save_weights(self.params.full_path_of_weights)
            print("Interrupted by KEYBOARD!", flush=True)
            print("saved weights to: {}".format(self.params.full_path_of_weights), flush=True)
            f.close()           # Close the file handle            

        end_train_session = time.time()
        f.close()               # Close the file handle

        # Safe Model parameters to .h5 file after training.
        self.model.save_weights(self.params.full_path_of_weights)
        print("\nSaved weights to: {}".format(self.params.full_path_of_weights), flush=True)
        print("\nSaved results to: {}".format(self.params.full_path_of_history), flush=True)
        print("\nTotal time employed ", hms(end_train_session - begin_train_session), flush=True)


    def format_info_for_csv(self, chunk_idx, history, begin_train_session):
        return [str(chunk_idx),
                str(history.history["loss"][0]),
                str(history.history["binary_accuracy"][0]),
                str(history.history["val_loss"][0]),
                str(history.history["val_binary_accuracy"][0]),
                str(hms(time.time()-begin_train_session)),
                str(psutil.cpu_percent()),
                str(psutil.virtual_memory().percent),
                str(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)]
            

    # Resets the backend of keras. Everything regarding the model is stored into a folder,
    # and later loaded again. This function is needed, in order to resolve memory leaks.
    # Keras 1.x does not show memory leaks, while keras 2.x does show memory leaks.
    # Resetting the backend of keras solves this problem. However this can be time
    # consuming when models get large.
    def reset_keras_backend(self):
        begin_time = time.time()
        print("\n\n----Reseting tensorflow keras backend", flush=True)
        self.model.save(self.params.full_path_model_storage)
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model(self.params.full_path_model_storage)
        print("\n reset time: {}\n----".format(hms(time.time() - begin_time)), flush=True)


    # Update network properties, based on the history of the trained network.
    def update_loss_and_acc(self, history):
        self.loss.append(history.history["loss"][0])
        self.acc.append(history.history["binary_accuracy"][0])
        self.val_loss.append(history.history["val_loss"][0])
        self.val_acc.append(history.history["val_binary_accuracy"][0])


    # Store Neural Network summary to file
    def save_model_to_file(self):
        with open(self.params.full_path_neural_net_printout,'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


    # Builds a Residual Neural Network with network depth 50
    def build_resnet50(self, input_shape = (101, 101, 1), num_outputs = 1):
    
        # Define the input as a Tensor with shape input_shape
        X_input = Input(input_shape)
        
        # Zero-Padding
        X = ZeroPadding2D((3,3))(X_input)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
        # Stage 2
        X = self.convolutional_block(X, f_size=3, filters = [64, 64, 256], stage=2, block='a', stride=1)
        X = self.identity_mapping_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_mapping_block(X, 3, [64, 64, 256], stage=2, block='c')
    
        # Stage 3
        X = self.convolutional_block(X, f_size=3, filters=[128, 128, 512], stage=3, block='a', stride=2)
        X = self.identity_mapping_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_mapping_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_mapping_block(X, 3, [128, 128, 512], stage=3, block='d')
    
        # Stage 4
        X = self.convolutional_block(X, f_size=3, filters=[256, 256, 1024], stage=4, block='a', stride=2)
        X = self.identity_mapping_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_mapping_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_mapping_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_mapping_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_mapping_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
        # Stage 5
        X = self.convolutional_block(X, f_size=3, filters=[512, 512, 2048], stage=5, block='a', stride=2)
        X = self.identity_mapping_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_mapping_block(X, 3, [512, 512, 2048], stage=5, block='c')
    
        # AVGPOOL
        if self.params.use_avg_pooling_2D:
            X = AveragePooling2D(pool_size=(2,2), padding='same')(X)
    
        # Output layer
        X = Flatten()(X)
        X = Dense(num_outputs, activation='sigmoid', name='fc_' + str(num_outputs), kernel_initializer = glorot_uniform(seed=0))(X)   # Glorot Uniform should be quite equivalent to He Uniform.
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='resnet50')

        # Compile the Model before returning it.
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        print(model.summary(), flush=True)
        return model
        
    
    # Define a residual block with three convoluational layers. Where the middle convolutional layer has a kernel size of fxf
    def identity_mapping_block(self, inp, f_size, filters, stage, block):
        
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
                   padding = 'valid',
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
                   padding = 'valid',
                   name = conv_name_base + '2c',
                   kernel_initializer = glorot_uniform(seed=0))(inp)
        inp = BatchNormalization(axis = 3, name = bn_name_base + '2c')(inp)   # Axis three is color channel
        
        # Step 4 - Add the shortcut to the main path before the relu activation function
        inp = Add()([inp, inp_shortcut])
        inp = Activation('relu')(inp)
        
        return inp
    
    
    # This convolutional block is a residual block but can be used with strides aswell. In order to perform downsampling.
    def convolutional_block(self, inp, f_size, filters, stage, block, stride=2):
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        # Unpack filters. filters is a list of filter amounts per step. (This block contains three steps of the main path, therefore there are three filter amounts.)
        nb_filters1, nb_filters2, nb_filters3 = filters
    
        # Save the input value
        inp_shortcut = inp
    
        # Step 1 - main path  - This Step has a stride > 1
        inp = Conv2D(filters=nb_filters1, kernel_size=(1, 1), strides=(stride, stride), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(inp)
        inp = BatchNormalization(axis=3, name=bn_name_base + '2a')(inp)   # Axis three is color channel
        inp = Activation('relu')(inp)
    
        # Step 2 - main path
        inp = Conv2D(filters=nb_filters2, kernel_size=(f_size, f_size), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(inp)
        inp = BatchNormalization(axis=3, name=bn_name_base + '2b')(inp)   # Axis three is color channel
        inp = Activation('relu')(inp)
    
        # Step 3 - main path
        inp = Conv2D(filters=nb_filters3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(inp)
        inp = BatchNormalization(axis=3, name=bn_name_base + '2c')(inp)   # Axis three is color channel
    
        # Step 4 - Shortcut connection   - This Step has a stride > 1
        inp_shortcut = Conv2D(filters=nb_filters3, kernel_size=(1, 1), strides=(stride, stride), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(inp_shortcut)
        inp_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(inp_shortcut)   # Axis three is color channel
    
        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        inp = Add()([inp, inp_shortcut])
        inp = Activation('relu')(inp)
    
        return inp


    def basic_block(self, filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        """

        def f(input):

            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv1 = Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    strides=init_strides,
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(1e-4),
                )(input)
            else:
                conv1 = self.bn_relu_conv(
                    filters=filters, kernel_size=(3, 3), strides=init_strides
                )(input)

            residual = self.bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
            return self.shortcut(input, residual)

        return f


    def shortcut(self, input, residual):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[self.row_axis] / residual_shape[self.row_axis]))
        stride_height = int(round(input_shape[self.col_axis] / residual_shape[self.col_axis]))
        equal_channels = input_shape[self.channel_axis] == residual_shape[self.channel_axis]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(
                filters=residual_shape[self.channel_axis],
                kernel_size=(1, 1),
                strides=(stride_width, stride_height),
                padding="valid",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(0.0001),
            )(input)

        return Add()([shortcut, residual])


    def bn_relu_conv(self, **conv_params):
        """Helper to build a BN -> relu -> conv block.
        This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.0e-4))

        def f(input):
            activation = self.bn_relu(input)
            return Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )(activation)

        return f


    def get_block(self, identifier):
        if isinstance(identifier, six.string_types):
            res = globals().get(identifier)
            if not res:
                raise ValueError("Invalid {}".format(identifier))
            return res
        return identifier


    def conv_bn_relu(self, **conv_params):
        """Helper to build a conv -> BN -> relu block
        """
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.0e-4))

        def f(input):
            conv = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )(input)
            return self.bn_relu(conv)

        return f



    def residual_block(self, block_function, filters, repetitions, is_first_layer=False):
        """Builds a residual block with repeating bottleneck blocks.
        """
        def f(input):
            for i in range(repetitions):
                init_strides = (1, 1)
                if i == 0 and not is_first_layer:
                    init_strides = (2, 2)
                input = block_function(
                    filters=filters,
                    init_strides=init_strides,
                    is_first_block_of_first_layer=(is_first_layer and i == 0),
                )(input)
            return input

        return f


    
    def bn_relu(self, input):
        """Helper to build a BN -> relu block
        """
        norm = BatchNormalization(axis=self.channel_axis)(input)
        return Activation("relu")(norm)


    def build_resnet(self, input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        # _handle_dim_ordering()
        # if len(input_shape) != 3:
        #    raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")
        #
        ## Permute dimension order if necessary
        # if K.image_dim_ordering() == 'tf':
        #    input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = self.get_block(block_fn)

        input = Input(shape=input_shape)
        # Gauss = GaussianNoise(0.01)(input)
        conv1 = self.conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = self.residual_block(
                block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0)
            )(block)
            filters *= 2

        # Last activation
        block = self.bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        if self.params.use_avg_pooling_2D:
            pool2 = AveragePooling2D(pool_size=(block_shape[self.row_axis], block_shape[self.col_axis]), strides=(1, 1))(block)
            flatten1 = Flatten()(pool2)
        else:
            flatten1 = Flatten()(block)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal", activation="sigmoid")(flatten1)

        model = Model(inputs=input, outputs=dense)
        print(model.summary(), flush=True)

        model.compile(
                        optimizer=self.optimizer,
                        loss=self.loss,
                        metrics=self.metrics,
                    )

        return model



    