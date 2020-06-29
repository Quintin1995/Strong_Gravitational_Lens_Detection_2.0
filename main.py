from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from load_data import *
from utils import *
import random
import matplotlib.pyplot as plt
from Network import *
from Parameters import *
from Parameters import Parameters

###### Step 0.1: Load all settings from .yaml file
settings_yaml = load_settings_yaml("runs/run.yaml")
params = Parameters(settings_yaml)
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.


###### Step 0.2: Create Directories used throughout the project
create_dir_if_not_exists("models")
create_dir_if_not_exists("runs")
create_dir_if_not_exists("slurms")


###### Step 1.0: Create and store normalized data arrays.
lenses_array    = get_data_array(params.img_dims, path=params.lenses_path, fraction_to_load=params.fraction_to_load_lenses, are_sources=False, normalize_dat=params.normalize)
negatives_array = get_data_array(params.img_dims, path=params.negatives_path, fraction_to_load=params.fraction_to_load_negatives, are_sources=False, normalize_dat=params.normalize)
sources_array   = get_data_array(params.img_dims, path=params.sources_path, fraction_to_load=params.fraction_to_load_sources, are_sources=True, normalize_dat=params.normalize)


###### Step 1.1: show some of the stored images from the data array to the user.
if params.verbatim:
    show_random_img_plt_and_stats(lenses_array,    num_imgs=1, title="lenses")
    show_random_img_plt_and_stats(negatives_array, num_imgs=1, title="negatives")
    show_random_img_plt_and_stats(sources_array,   num_imgs=1, title="sources")


###### Step 2: Split the data into train and test data.
X_train_lenses, X_test_lenses, y_train_lenses, y_test_lenses             = split_train_test_data(lenses_array, 0.0, test_fraction=params.test_fraction)
X_train_negatives, X_test_negatives, y_train_negatives, y_test_negatives = split_train_test_data(negatives_array, 0.0, test_fraction=params.test_fraction)
X_train_sources, X_test_sources, _, _                                    = split_train_test_data(sources_array, "no_label",  test_fraction=params.test_fraction)


###### Step 4.0 - Load a Training Chunk of data - This includes the merging of lenses with sources (creation of mock lenses).
X_train_chunk, y_train_chunk = load_chunk(params.chunksize, X_train_lenses, X_train_negatives, X_train_sources, params.data_type, params.mock_lens_alpha_scaling)
X_test_chunk, y_test_chunk   = load_chunk(params.chunksize, X_test_lenses, X_test_negatives, X_test_sources, params.data_type, params.mock_lens_alpha_scaling)


###### Step 4.1 - Sanity check of the train and test chunk
# 1: Are both positive and negative examples within the same brightness ranges?
# 2: I have added a per image normalization, because a couple of outliers ruin the normalization per data array (That is my hypothesis at least.)
if False:
    print(y_train_chunk)
    idxs_pos = np.where(y_train_chunk == 1.0)
    idxs_neg = np.where(y_train_chunk == 0.0)

    for i in range(25):
        pos_img = X_train_chunk[random.choice(list(idxs_pos[0]))]
        neg_img = X_train_chunk[random.choice(list(idxs_neg[0]))]
        show2Imgs(pos_img, neg_img, "pos max pixel: {0:.3f}".format(np.amax(pos_img)), "neg max pixel: {0:.3f}".format(np.amax(neg_img)))


###### Step 5.0 - Data Augmentation - Data Generator Keras - Training Generator is based on train data array.
train_generator = ImageDataGenerator(
        rotation_range=params.aug_rotation_range,
        width_shift_range=params.aug_width_shift_range,
        height_shift_range=params.aug_height_shift_range,
        zoom_range=params.aug_zoom_range,
        horizontal_flip=params.aug_do_horizontal_flip,
        fill_mode=params.aug_default_fill_mode
        )


###### Step 5.1 - Data Augmentation - Data Generator Keras - Validation Generator is based on test data for now
validation_generator = ImageDataGenerator(
        # rotation_range=params.aug_rotation_range,
        # width_shift_range=params.aug_width_shift_range,
        # height_shift_range=params.aug_height_shift_range,
        # zoom_range=params.aug_zoom_range,
        horizontal_flip=params.aug_do_horizontal_flip,
        fill_mode=params.aug_default_fill_mode
        )


###### Step 6.0 - Create Neural Network - Resnet18
resnet18   = Network(params.net_name, params.net_learning_rate, params.net_model_metrics, params.img_dims, params.net_num_outputs)


###### Step 7.0 - Init list for storing results of network
loss     = []              # Store loss of the model
acc      = []              # Store binary accuracy of the model
val_loss = []              # Store validation loss of the model
val_acc  = []              # Store validation binary accuracy


###### Step 7.1 - Training the model
begin_train_session = time.time()       # Records beginning of training time
try:
    for chunk_idx in range(params.num_chunks):
        print("chunk {}/{}".format(chunk_idx+1, params.num_chunks))

        # Load chunk
        X_train_chunk, y_train_chunk = load_chunk(params.chunksize, X_train_lenses, X_train_negatives, X_train_sources, params.data_type, params.mock_lens_alpha_scaling)
        
        # Fit model on data with a keras image data generator
        network_fit_time_start = time.time()
        # history = resnet18.model.fit_generator(train_generator.flow(    X_train_chunk,
        #                                                                 y_train_chunk,
        #                                                                 batch_size=params.net_batch_size),
        #                                                                 steps_per_epoch=len(X_train_chunk) / params.net_batch_size,
        #                                                                 epochs=params.net_epochs)

        # Define a train generator flow based on the ImageDataGenerator
        train_generator_flowed = train_generator.flow(
            X_train_chunk,
            y_train_chunk,
            batch_size=params.net_batch_size)

        # Define a validation generator flow based on the ImageDataGenerator
        validation_generator_flowed = validation_generator.flow(
            X_test_chunk,
            y_test_chunk,
            batch_size=params.net_batch_size)

        # Fit both generators (train and validation) on the model.
        history = resnet18.model.fit_generator(
                train_generator_flowed,
                steps_per_epoch=len(X_train_chunk) / params.net_batch_size,
                epochs=params.net_epochs,
                validation_data=validation_generator_flowed,
                validation_steps=params.validation_steps)

        print("Training on chunk took: {}".format(hms(time.time() - network_fit_time_start)))

        # Save Model params to .h5 file
        if chunk_idx % params.chunk_save_interval == 0:
            resnet18.model.save_weights(params.full_path_of_weights)
            print("Saved model weights to: {}".format(params.full_path_of_weights))

        # Store loss and accuracy in list
        loss.append(history.history["loss"][0])
        acc.append(history.history["binary_accuracy"][0])
        val_loss.append(history.history["val_loss"][0])
        val_acc.append(history.history["val_binary_accuracy"][0])
        
        # Plot loss and accuracy on interval (also validation loss and accuracy) (to a png file)
        if chunk_idx % params.chunk_plot_interval == 0:
            plot_history(acc, val_acc, loss, val_loss, params)

except KeyboardInterrupt:
    resnet18.model.save_weights(params.full_path_of_weights)
    print("Interrupted by KEYBOARD!", flush=True)
    print("saved weights to: {}".format(params.full_path_of_weights), flush=True)

end_train_session = time.time()

# Safe Model parameters to .h5 file after training.
resnet18.model.save_weights(params.full_path_of_weights)
print("\nSaved weights to: {}".format(params.full_path_of_weights), flush=True)
print("\nSaved results to: {}".format(params.full_path_of_history), flush=True)
print("\nTotal time employed ",hms(end_train_session - begin_train_session), flush=True)
