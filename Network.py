from __future__ import division

import six
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, GaussianNoise
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import csv
import time
from utils import *


class Network:
    def __init__(self, params, data_generator):
        print("Initializing Network.")
        self.params        = params
        self.data_gen      = data_generator

        self.row_axis      = 1
        self.col_axis      = 2
        self.channel_axis  = 3
        self.name          = self.params.net_name
        self.learning_rate = self.params.net_learning_rate
        self.optimizer     = optimizers.Adam(lr=self.learning_rate)
        self.loss          = "binary_crossentropy"
        self.input_shape   = self.params.img_dims
        self.num_outputs   = self.params.net_num_outputs
        self.metrics       = [metrics.binary_accuracy] if self.params.net_model_metrics == "binary_accuracy" else None

        if self.name == "resnet18":
            self.model = self.build_resnet(self.input_shape, self.num_outputs, self.basic_block, [2, 2, 2, 2])

        if self.model == None:
            print("Model of the network class is not set.")

        # Define an ImageDataGenerator that will augment the data in real time as a generator (training) - DATA AUGMENTATION
        self.train_generator = ImageDataGenerator(
                rotation_range=self.params.aug_rotation_range,
                width_shift_range=self.params.aug_width_shift_range,
                height_shift_range=self.params.aug_height_shift_range,
                zoom_range=self.params.aug_zoom_range,
                horizontal_flip=self.params.aug_do_horizontal_flip,
                fill_mode=self.params.aug_default_fill_mode
                )

        # Define an ImageDataGenerator that will augment the data in real time as a generator (validation) - DATA AUGMENTATION
        self.validation_generator = ImageDataGenerator(
                horizontal_flip=self.params.aug_do_horizontal_flip,
                fill_mode=self.params.aug_default_fill_mode
                )
        print("Done Initializing Network.")


    ######## BEGIN NETWORK TRAIN METHODS ########
    def train_resnet18(self):
        loss     = []              # Store loss of the model
        acc      = []              # Store binary accuracy of the model
        val_loss = []              # Store validation loss of the model
        val_acc  = []              # Store validation binary accuracy

        ###### Step 7.1 - Training the model
        # Create a csv logger that will store the history of the .fit function into a .csv file
        with open(self.params.full_path_of_history, 'w', newline='') as history_file:
            writer = csv.writer(history_file)
            writer.writerow(["chunk", "loss", "binary_accuracy", "val_loss", "val_binary_accuracy"])

            begin_train_session = time.time()       # Records beginning of training time

            try:
                for chunk_idx in range(self.params.num_chunks):
                    print("chunk {}/{}".format(chunk_idx+1, self.params.num_chunks), flush=True)

                    # Load chunk - train
                    X_train_chunk, y_train_chunk = self.data_gen.load_chunk(self.params.chunksize,
                                                                            self.data_gen.X_train_lenses,
                                                                            self.data_gen.X_train_negatives,
                                                                            self.data_gen.X_train_sources,
                                                                            self.params.data_type,
                                                                            self.params.mock_lens_alpha_scaling)
                    # Load chunk - validation - (test)
                    X_test_chunk, y_test_chunk = self.data_gen.load_chunk(self.params.validation_steps,
                                                                            self.data_gen.X_test_lenses,
                                                                            self.data_gen.X_test_negatives,
                                                                            self.data_gen.X_test_sources,
                                                                            self.params.data_type,
                                                                            self.params.mock_lens_alpha_scaling)
                    # Record beginning of chunk train time
                    network_fit_time_start = time.time()

                    # Define a train generator flow based on the ImageDataGenerator
                    train_generator_flowed = self.train_generator.flow(
                        X_train_chunk,
                        y_train_chunk,
                        batch_size=self.params.net_batch_size)

                    # Define a validation generator flow based on the ImageDataGenerator
                    validation_generator_flowed = self.validation_generator.flow(
                        X_test_chunk,
                        y_test_chunk,
                        batch_size=self.params.net_batch_size)

                    # Fit both generators (train and validation) on the model.
                    history = self.model.fit_generator(
                            train_generator_flowed,
                            steps_per_epoch=len(X_train_chunk) / self.params.net_batch_size,
                            epochs=self.params.net_epochs,
                            validation_data=validation_generator_flowed,
                            validation_steps=self.params.validation_steps)

                    print("Training on chunk took: {}".format(hms(time.time() - network_fit_time_start)), flush=True)

                    # Save Model self.params to .h5 file
                    if chunk_idx % self.params.chunk_save_interval == 0:
                        self.model.save_weights(self.params.full_path_of_weights)
                        print("Saved model weights to: {}".format(self.params.full_path_of_weights),flush=True)

                    # Write results to csv for later use
                    writer.writerow([str(chunk_idx), str(history.history["loss"][0]), str(history.history["binary_accuracy"][0]), str(history.history["val_loss"][0]), str(history.history["val_binary_accuracy"][0])])

                    # Store loss and accuracy in list
                    loss.append(history.history["loss"][0])
                    acc.append(history.history["binary_accuracy"][0])
                    val_loss.append(history.history["val_loss"][0])
                    val_acc.append(history.history["val_binary_accuracy"][0])
                    
                    # Plot loss and accuracy on interval (also validation loss and accuracy) (to a png file)
                    if chunk_idx % self.params.chunk_plot_interval == 0:
                        plot_history(acc, val_acc, loss, val_loss, self.params)

            except KeyboardInterrupt:
                self.model.save_weights(self.params.full_path_of_weights)
                print("Interrupted by KEYBOARD!", flush=True)
                print("saved weights to: {}".format(self.params.full_path_of_weights), flush=True)
                print("\nTotal time employed ",hms(time.time(0) - begin_train_session), flush=True)
                plot_history(acc, val_acc, loss, val_loss, self.params)

            end_train_session = time.time()

            # Safe Model parameters to .h5 file after training.
            self.model.save_weights(self.params.full_path_of_weights)
            print("\nSaved weights to: {}".format(self.params.full_path_of_weights), flush=True)
            print("\nSaved results to: {}".format(self.params.full_path_of_history), flush=True)
            print("\nTotal time employed ",hms(end_train_session - begin_train_session), flush=True)
            plot_history(acc, val_acc, loss, val_loss, self.params)
    ######## END NETWORK TRAIN METHODS   ########

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

        return add([shortcut, residual])


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
        pool2 = AveragePooling2D(pool_size=(block_shape[self.row_axis], block_shape[self.col_axis]), strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal", activation="sigmoid")(flatten1)

        model = Model(inputs=input, outputs=dense)
        print(model.summary(), flush=True)

        model.compile(
                        optimizer=self.optimizer,
                        loss=self.loss,
                        metrics=self.metrics,
                    )

        return model



    