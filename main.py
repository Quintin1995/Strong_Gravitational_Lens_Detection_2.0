import numpy as np
from utils import *
import random
import matplotlib.pyplot as plt
from Network import *
from Parameters import *
from Parameters import Parameters
from DataGenerator import *
import csv
import psutil


###### Step 1.0: Load all settings from .yaml file
settings_yaml = load_settings_yaml("runs/run.yaml")
params = Parameters(settings_yaml)
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.


###### Step 2.0 Create Image Data Generator
dg = DataGenerator(params)


###### Step 3.0 - Create Neural Network - Resnet18
resnet18 = Network(params.net_name, params.net_learning_rate, params.net_model_metrics, params.img_dims, params.net_num_outputs, params)


###### Step 4.0 - Training
loss     = []              # Store loss of the model
acc      = []              # Store binary accuracy of the model
val_loss = []              # Store validation loss of the model
val_acc  = []              # Store validation binary accuracy

# with open(params.full_path_of_history, 'w', newline='') as history_file:
bufsize = 1
f = open(params.full_path_of_history, "w", bufsize)
writer = csv.writer(f)
writer.writerow(["chunk", "loss", "binary_accuracy", "val_loss", "val_binary_accuracy", "time", "cpu_percentage", "ram_usage", "available_mem"])

###### Step 7.1 - Training the model
begin_train_session = time.time()       # Records beginning of training time
try:
    for chunk_idx in range(params.num_chunks):
        print("chunk {}/{}".format(chunk_idx+1, params.num_chunks), flush=True)

        # Load chunk
        X_train_chunk, y_train_chunk = dg.load_chunk(params.chunksize, dg.Xlenses_train, dg.Xnegatives_train, dg.Xsources_train, params.data_type, params.mock_lens_alpha_scaling)
        X_validation_chunk, y_validation_chunk = dg.load_chunk(params.validation_chunksize, dg.Xlenses_validation, dg.Xnegatives_validation, dg.Xsources_validation, params.data_type, params.mock_lens_alpha_scaling)

        # Fit model on data with a keras image data generator
        network_fit_time_start = time.time()

        # Define a train generator flow based on the ImageDataGenerator
        train_generator_flowed = dg.train_generator.flow(
            X_train_chunk,
            y_train_chunk,
            batch_size=params.net_batch_size)

        # Define a validation generator flow based on the ImageDataGenerator
        validation_generator_flowed = dg.validation_generator.flow(
            X_validation_chunk,
            y_validation_chunk,
            batch_size=params.net_batch_size)

        # Fit both generators (train and validation) on the model.
        history = resnet18.model.fit_generator(
                train_generator_flowed,
                steps_per_epoch=len(X_train_chunk) / params.net_batch_size,
                epochs=params.net_epochs,
                validation_data=validation_generator_flowed,
                validation_steps=len(X_validation_chunk) / params.net_batch_size)

        print("Training on chunk took: {}".format(hms(time.time() - network_fit_time_start)), flush=True)

        # Try to reset data
        X_train_chunk = None
        y_train_chunk = None
        X_validation_chunk = None
        y_validation_chunk = None
        train_generator_flowed = None
        validation_generator_flowed = None

        # Save Model params to .h5 file
        if chunk_idx % params.chunk_save_interval == 0:
            resnet18.model.save_weights(params.full_path_of_weights)
            print("Saved model weights to: {}".format(params.full_path_of_weights), flush=True)

        # Write results to csv for later use
        writer.writerow([str(chunk_idx),
                        str(history.history["loss"][0]),
                        str(history.history["binary_accuracy"][0]),
                        str(history.history["val_loss"][0]),
                        str(history.history["val_binary_accuracy"][0]),
                        str(hms(time.time()-begin_train_session)),
                        str(psutil.cpu_percent()),
                        str(psutil.virtual_memory().percent),
                        str(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)])

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
    f.close()

end_train_session = time.time()
f.close()

# Safe Model parameters to .h5 file after training.
resnet18.model.save_weights(params.full_path_of_weights)
print("\nSaved weights to: {}".format(params.full_path_of_weights), flush=True)
print("\nSaved results to: {}".format(params.full_path_of_history), flush=True)
print("\nTotal time employed ",hms(end_train_session - begin_train_session), flush=True)
