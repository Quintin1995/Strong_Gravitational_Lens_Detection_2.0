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
        
        # Setting the loss function
        self.metrics = None
        self.set_neural_network_metric()

        # Neural Network Train Metric
        self.loss_function = None
        self.set_loss_function()
        
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
        elif self.params.net_name == "resnet50":
            self.model          = self.build_resnet50(input_shape = self.input_shape, num_outputs = self.num_outputs)

        # Parameters used when training the model
        self.loss     = []              # Store loss of the model
        self.acc      = []              # Store binary accuracy of the model        or it holds other metric values such as f1-softloss (scores)
        self.val_loss = []              # Store validation loss of the model
        self.val_acc  = []              # Store validation binary accuracy           or it holds other metric values such as f1-softloss (scores)

        # Model Selection
        self.best_val_loss    = 9999.0

        # Early Stopping
        self.es_patience      = self.params.es_patience
        self.patience_counter = 0

        # Validaton -  After this amount of epochs/train-chunks validation will be performed
        self.net_val_freq     = self.params.net_val_freq      

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

        if self.metrics == "binary_accuracy":
            writer.writerow(["chunk", "loss", "binary_accuracy", "val_loss", "val_binary_accuracy", "time", "cpu_percentage", "ram_usage", "available_mem"])
        else:
            writer.writerow(["chunk", "loss", "macro_f1", "val_loss", "val_macro_f1", "time", "cpu_percentage", "ram_usage", "available_mem"])

        # Train the model
        begin_train_session = time.time()       # Records beginning of training time
        try:
            for chunk_idx in range(self.params.num_chunks):
                
                print("chunk {}/{}".format(chunk_idx+1, self.params.num_chunks), flush=True)

                # Load train chunk and targets
                X_train_chunk, y_train_chunk = self.dg.load_chunk(self.params.chunksize, self.dg.Xlenses_train, self.dg.Xnegatives_train, self.dg.Xsources_train, self.params.data_type, self.params.mock_lens_alpha_scaling)
                # Load validation chunk and targets
                # X_validation_chunk, y_validation_chunk = self.dg.load_chunk(self.params.validation_chunksize, self.dg.Xlenses_validation, self.dg.Xnegatives_validation, self.dg.Xsources_validation, self.params.data_type, self.params.mock_lens_alpha_scaling)
                if chunk_idx % self.net_val_freq == 0:
                    X_validation_chunk, y_validation_chunk = self.dg.load_chunk_val(data_type=np.float32, mock_lens_alpha_scaling=self.params.mock_lens_alpha_scaling)

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
                        validation_steps=len(X_validation_chunk) / self.params.net_batch_size,
                        validation_freq=self.net_val_freq)

                print("Training on chunk took: {}".format(hms(time.time() - network_fit_time_start)), flush=True)

                # Write results to csv file
                writer.writerow(self.format_info_for_csv(chunk_idx, history, begin_train_session))

                # Store loss and accuracy in list
                self.update_loss_and_acc(history)
                
                # Plot loss and accuracy on interval (also validation loss and accuracy) (to a png file)
                if chunk_idx % self.params.chunk_plot_interval == 0:
                    plot_history(self.acc, self.val_acc, self.loss, self.val_loss, self.params)

                # MODEL SELECTION
                # It seems reasonalbe to assume that we don't want to store all the early models, due to having a lower validation score.
                # Not storing the network before 4000 chunks have been trained on, seems like a reasonable efficiency heuristic.
                if chunk_idx > 4000 and history.history["val_loss"][0] < self.best_val_loss:    
                    self.model.save_weights(self.params.full_path_of_weights)
                    self.best_val_loss = history.history["val_loss"][0]
                    print("better validation: Saved model weights to: {}".format(self.params.full_path_of_weights), flush=True)

                # Reset backend of tensorflow so that memory does not leak - Clear keras backend when at 75% usage.
                if(psutil.virtual_memory().percent > 75.0):
                    self.reset_keras_backend()

                # Early stopping
                if self.do_early_stopping(history.history["val_loss"][0]):
                    break

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


    # Returns True if early stopping condition has been met.
    def do_early_stopping(self, vall_loss):
        if vall_loss < min(self.val_loss):
            self.patience_counter = 0
        else:                              # The model did not improve
            self.patience_counter += 1          
        if self.patience_counter == self.es_patience:
            # Stop Training
            print("Early Stopping!!!", flush=True)
            return True
        return False


    def format_info_for_csv(self, chunk_idx, history, begin_train_session):
        if self.metrics == "binary_accuracy":
            return [str(chunk_idx),
                    str(history.history["loss"][0]),
                    str(history.history["binary_accuracy"][0]),
                    str(history.history["val_loss"][0]),
                    str(history.history["val_binary_accuracy"][0]),
                    str(hms(time.time()-begin_train_session)),
                    str(psutil.cpu_percent()),
                    str(psutil.virtual_memory().percent),
                    str(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)]
        elif self.metrics == self.macro_f1:
            return [str(chunk_idx),
                    str(history.history["loss"][0]),
                    str(history.history["macro_f1"][0]),
                    str(history.history["val_loss"][0]),
                    str(history.history["val_macro_f1"][0]),
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
        self.model = tf.keras.models.load_model(self.params.full_path_model_storage, compile=False)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=[self.metrics])
        print("\n reset time: {}\n----".format(hms(time.time() - begin_time)), flush=True)


    # Update network properties, based on the history of the trained network.
    def update_loss_and_acc(self, history):
        if self.metrics == "binary_accuracy":
            self.loss.append(history.history["loss"][0])
            self.acc.append(history.history["binary_accuracy"][0])
            self.val_loss.append(history.history["val_loss"][0])
            self.val_acc.append(history.history["val_binary_accuracy"][0])
        else:
            self.loss.append(history.history["loss"][0])
            self.acc.append(history.history["macro_f1"][0])
            self.val_loss.append(history.history["val_loss"][0])
            self.val_acc.append(history.history["val_macro_f1"][0])


    # Store Neural Network summary to file
    def save_model_to_file(self):
        with open(self.params.full_path_neural_net_printout,'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


    # Case over all possible loss functions
    def set_loss_function(self):
        if self.params.net_loss_function == "binary_crossentropy":
            self.loss_function = "binary_crossentropy"
        elif self.params.net_loss_function == "macro_soft_f1":
            self.loss_function = self.macro_soft_f1
        elif self.params.net_loss_function == "macro_double_soft_f1":
            self.loss_function = self.macro_double_soft_f1
        else:
            print("No valid loss function has been selected.")
            self.loss_function = None


    # Case over all possible training metrics
    def set_neural_network_metric(self):
        if self.params.net_model_metrics == "binary_accuracy":
            self.metrics = "binary_accuracy"
        elif self.params.net_model_metrics == "macro_f1":
            self.metrics = self.macro_f1
        else:
            self.metrics = None


    # The following function/method I have taken from:
    # https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    # Written by: Ashref Maiza
    def macro_soft_f1(self, y, y_hat):
        """Compute the macro soft F1-score as a cost.
        Average (1 - soft-F1) across all labels.
        Use probability values instead of binary predictions.
        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = tf.reduce_mean(cost) # average on all labels
        return macro_cost


    # The following function/method I have taken from:
    # https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    # Written by: Ashref Maiza
    def macro_double_soft_f1(self, y, y_hat):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.
        
        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
        soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
        soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
        cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
        macro_cost = tf.reduce_mean(cost) # average on all labels
        return macro_cost


    # The following function/method I have taken from:
    # https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    # Written by: Ashref Maiza
    def macro_f1(self, y, y_hat, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        macro_f1 = tf.reduce_mean(f1)
        return macro_f1


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
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=[self.metrics])
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
                        loss=self.loss_function,
                        metrics=[self.metrics],
                    )

        return model



    