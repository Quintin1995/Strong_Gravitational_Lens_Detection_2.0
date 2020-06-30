import numpy as np
from utils import *
import random
import matplotlib.pyplot as plt
from Network import *
from Parameters import *
from Parameters import Parameters
from DataGenerator import *
import csv


###### Step 0.1: Load all settings from .yaml file
settings_yaml = load_settings_yaml("runs/run.yaml")
params = Parameters(settings_yaml)
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.


dg = DataGenerator(params)

###### Step 1.1: show some of the stored images from the data array to the user.
if params.verbatim:
    show_random_img_plt_and_stats(lenses_array,    num_imgs=1, title="lenses")
    show_random_img_plt_and_stats(negatives_array, num_imgs=1, title="negatives")
    show_random_img_plt_and_stats(sources_array,   num_imgs=1, title="sources")


###### Step 4.1 - Sanity check of the train and test chunk
# 1: Are both positive and negative examples within the same brightness ranges?
# 2: I have added a per image normalization, because a couple of outliers ruin the normalization per data array (That is my hypothesis at least.)
if params.verbatim:
    print(y_train_chunk)
    idxs_pos = np.where(y_train_chunk == 1.0)
    idxs_neg = np.where(y_train_chunk == 0.0)

    for i in range(25):
        pos_img = X_train_chunk[random.choice(list(idxs_pos[0]))]
        neg_img = X_train_chunk[random.choice(list(idxs_neg[0]))]
        show2Imgs(pos_img, neg_img, "pos max pixel: {0:.3f}".format(np.amax(pos_img)), "neg max pixel: {0:.3f}".format(np.amax(neg_img)))




###### Step 6.0 - Create Neural Network - Resnet18
resnet18   = Network(params.net_name, params.net_learning_rate, params.net_model_metrics, params.img_dims, params.net_num_outputs, params)


###### Step 7.0 - Init list for storing results of network
loss     = []              # Store loss of the model
acc      = []              # Store binary accuracy of the model
val_loss = []              # Store validation loss of the model
val_acc  = []              # Store validation binary accuracy

with open(params.full_path_of_history, 'w', newline='') as history_file:
    writer = csv.writer(history_file)
    writer.writerow(["chunk", "loss", "binary_accuracy", "val_loss", "val_binary_accuracy", "time"])

    ###### Step 7.1 - Training the model
    begin_train_session = time.time()       # Records beginning of training time
    try:
        for chunk_idx in range(params.num_chunks):
            print("chunk {}/{}".format(chunk_idx+1, params.num_chunks))

            # Load chunk
            X_train_chunk, y_train_chunk = dg.load_chunk(params.chunksize, dg.X_train_lenses, dg.X_train_negatives, dg.X_train_sources, params.data_type, params.mock_lens_alpha_scaling)
            X_test_chunk, y_test_chunk = dg.load_chunk(params.validation_steps, dg.X_test_lenses, dg.X_test_negatives, dg.X_test_sources, params.data_type, params.mock_lens_alpha_scaling)

            # Fit model on data with a keras image data generator
            network_fit_time_start = time.time()

            # Define a train generator flow based on the ImageDataGenerator
            train_generator_flowed = dg.train_generator.flow(
                X_train_chunk,
                y_train_chunk,
                batch_size=params.net_batch_size)

            # Define a validation generator flow based on the ImageDataGenerator
            validation_generator_flowed = dg.validation_generator.flow(
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

            # Write results to csv for later use
            writer.writerow([str(chunk_idx),
                            str(history.history["loss"][0]),
                            str(history.history["binary_accuracy"][0]),
                            str(history.history["val_loss"][0]),
                            str(history.history["val_binary_accuracy"][0]),
                            str(hms(time.time()-begin_train_session))])

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
