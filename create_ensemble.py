import argparse
from DataGenerator import DataGenerator
import glob
from Network import Network
import numpy as np
import os
from Parameters import Parameters
import tensorflow as tf
from utils import get_model_paths, get_h5_path_dialog, load_settings_yaml, binary_dialog, hms, load_normalize_img, get_samples, compute_PSF_r, normalize_img, create_dir_if_not_exists, dstack_data, count_TP_TN_FP_FN_and_FB
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import time
import random
from scipy.optimize import minimize
from scipy.special import softmax
from functools import partial
import yaml
import math
############################################################ Functions ############################################################


# Dump ensemble parameters to a yaml file as a dictionary
def _write_params_to_yaml(model_names, model_weights, args, ensemble_dir, acc, threshold, sample_size, precision, recall, fbeta):
    ensemble_dict = {
        "model_names":       model_names,
        "model_weights":     [float(str(x)) for x in model_weights],
        "ensemble_size":     len(model_names),
        "ensemble_method":   args.method,
        "accuracy":          acc,
        "threshold":         threshold,
        "sample_size":       sample_size,
        "precision":         precision,
        "recall":            recall,
        "f_beta":            fbeta

    }
    ensemble_params_fname = os.path.join(ensemble_dir, "ensemble_parameters.yaml")
    with open(ensemble_params_fname, 'w') as outfile:
        yaml.dump(ensemble_dict, outfile, default_flow_style=False)


# Calculate the Mean Square Error for a matrix of datapoints given a ground truth label
# and weight vector.
def mean_square_error(y, Xs, w0):

    # Prediction of the ensemble.
    y_hat = np.dot(Xs, w0)

    # Square the error in order to get rid of negative values (common practice)
    error = np.square(y - y_hat)

    # Take average error
    mean = np.mean(error)
    return mean

    
# Merge a single lens and source together into a mock lens.
def merge_lens_and_source(lens, source, mock_lens_alpha_scaling = (0.02, 0.30), show_imgs = False):

    # Add lens and source together | We rescale the brightness of the simulated source to the peak brightness of the LRG in the r-band multiplied by a factor of alpha randomly drawn from the interval [0.02,0.3]
    mock_lens = lens + source / np.amax(source) * np.amax(lens) * np.random.uniform(mock_lens_alpha_scaling[0], mock_lens_alpha_scaling[1])
    
    # Take a square root stretch to emphesize lower luminosity features.
    mock_lens = np.sqrt(mock_lens)
    
    # Basically removes negative values - should not be necessary, because all input data should be normalized anyway. (I will leave it for now, but should be removed soon.)
    # mock_lens_sqrt = mock_lens_sqrt.clip(min=0.0, max=1.0)
    mock_lens = mock_lens.clip(min=0.0, max=1.0)

    # if True:
    #     show2Imgs(lens, source, "Lens max pixel: {0:.3f}".format(np.amax(lens)), "Source max pixel: {0:.3f}".format(np.amax(source)))
        # show2Imgs(mock_lens, mock_lens_sqrt, "mock_lens max pixel: {0:.3f}".format(np.amax(mock_lens)), "mock_lens_sqrt max pixel: {0:.3f}".format(np.amax(mock_lens_sqrt)))

    return mock_lens


# This function should read images from the lenses- and sources data array,
# and merge them together into a lensing system, further described as 'mock lens'.
# These mock lenses represent a strong gravitational lensing system that should 
# get the label 1.0 (positive label). 
def merge_lenses_and_sources(lenses_array, sources_array, num_mock_lenses, data_type, mock_lens_alpha_scaling = (0.02, 0.30), do_deterministic=False):
    num_lenses  = lenses_array.shape[0]
    num_sources = sources_array.shape[0]

    X_train_positive = np.empty((num_mock_lenses, lenses_array.shape[1], lenses_array.shape[2], lenses_array.shape[3]), dtype=data_type)
    Y_train_positive = np.ones(num_mock_lenses, dtype=data_type)
    
    # Which indexes to load from sources and lenses (these will be merged together)
    if do_deterministic:
        idxs_lenses = list(range(num_lenses))       # if deterministic we always want the same set
        idxs_sources = random.choices(list(range(num_sources)), k=num_mock_lenses)
    else:
        idxs_lenses = random.choices(list(range(num_lenses)), k=num_mock_lenses)
        idxs_sources = random.choices(list(range(num_sources)), k=num_mock_lenses)

    for i in range(num_mock_lenses):
        lens   = lenses_array[idxs_lenses[i]]
        source = sources_array[idxs_sources[i]]
        mock_lens = merge_lens_and_source(lens, source, mock_lens_alpha_scaling)

        # Uncomment this code if you want to inspect how a lens, source and mock lens look before they are merged.
        # import matplotlib.pyplot as plt
        # l = np.squeeze(lens)
        # s = np.squeeze(source)
        # m = np.squeeze(mock_lens)
        # plt.clf()
        # plt.imshow(l, cmap='gray')
        # plt.show()
        # plt.imshow(s, cmap='gray')
        # plt.show()
        # plt.imshow(m, cmap='gray')
        # plt.show()

        X_train_positive[i] = mock_lens

    return X_train_positive, Y_train_positive


# Loading a valiation chunk into memory
def load_chunk_test(data_type, mock_lens_alpha_scaling, lenses_test, sources_test, negatives_test, seed=4321):

    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    num_positive = lenses_test.shape[0]
    
    # Lets create a balanced test set.
    num_negative = num_positive

    # Get mock lenses data and labels
    X_pos, y_pos = merge_lenses_and_sources(lenses_test, sources_test, num_positive, data_type, mock_lens_alpha_scaling, do_deterministic=True)
        
    # Store Negative data in numpy array and create label vector
    X_neg = np.empty((num_negative, X_pos.shape[1], X_pos.shape[2], X_pos.shape[3]), dtype=data_type)    
    y_neg = np.zeros(X_neg.shape[0], dtype=data_type)

    # We want a 80% probability of selecting from the contaminants set, and 20% probability of selecting an LRG from the lenses set.
    negative_sample_contaminant_prob = 0.8
    for i in range(num_negative):
        if random.random() <= negative_sample_contaminant_prob:
            X_neg[i] = np.sqrt(negatives_test[random.randint(0, negatives_test.shape[0] - 1)])        #sqrt stretch is needed, because the positive examples are also sqrt stretched
        else:
            X_neg[i] = np.sqrt(lenses_test[random.randint(0, lenses_test.shape[0] - 1)])        #sqrt stretch is needed, because the positive examples are also sqrt stretched

    # Concatenate the positive and negative examples into one chunk (also the labels)
    X_chunk = np.concatenate((X_pos, X_neg))
    y_chunk = np.concatenate((y_pos, y_neg))

    print("Creating test chunk took: {}, chunksize: {}".format(hms(time.time() - start_time), num_positive+num_negative), flush=True)
    return X_chunk, y_chunk


# Loading a valiation chunk into memory
def load_chunk_val(Xlenses_validation, Xsources_validation, Xnegatives_validation, mock_lens_alpha_scaling):
    start_time = time.time()
    num_positive = Xlenses_validation.shape[0]
    # num_negative = Xnegatives_validation.shape[0] + Xlenses_validation.shape[0]   #also num positive, because the unmerged lenses with sources are also deemed a negative sample.
    
    # Lets create a balanced validation set.
    num_negative = num_positive

    # Get mock lenses data and labels
    X_pos, y_pos = merge_lenses_and_sources(Xlenses_validation, Xsources_validation, num_positive, np.float32, mock_lens_alpha_scaling, do_deterministic=True)
        
    # Store Negative data in numpy array and create label vector
    X_neg = np.empty((num_negative, X_pos.shape[1], X_pos.shape[2], X_pos.shape[3]), dtype=np.float32)    
    y_neg = np.zeros(X_neg.shape[0], dtype=np.float32)

    # Negatives consist of the negatives set and the unmerged lenses set. A lens unmerged with a source is basically a negative.
    n = int(num_negative // 2)  # number samples to take from lenses-, and negatives set.
    
    indexes_lenses = np.random.choice(Xlenses_validation.shape[0], n, replace=False)  
    X_neg[0:int(num_negative//2)] = Xlenses_validation[indexes_lenses] # first half of negative chunk is a random selection from lenses without replacement

    indexes_negatives = np.random.choice(Xlenses_validation.shape[0], n+1, replace=False)  
    X_neg[int(num_negative//2):num_negative] = Xnegatives_validation[indexes_negatives] # second half of negatives are a random selection from the negatives without replacement
    
    # The negatives need a square root stretch, just like the positives.
    X_neg = np.sqrt(X_neg)

    # Concatenate the positive and negative examples into one chunk (also the labels)
    X_chunk = np.concatenate((X_pos, X_neg))
    y_chunk = np.concatenate((y_pos, y_neg))

    print("Creating validation chunk took: {}, chunksize: {}".format(hms(time.time() - start_time), num_positive+num_negative), flush=True)
    return X_chunk, y_chunk


# Choose multiple models as Ensemble members.
def choose_ensemble_members():
    model_paths = list()
    do_select_more_models = True
    while do_select_more_models:
        model_paths += get_model_paths()
        do_select_more_models = binary_dialog("\nDo you want to select more members for the Ensemble?")
    print("\nChoosen models: ")
    for model_path in model_paths:
        print("model path = {}".format(model_path))
    return model_paths



# Given 4D numpy image data and corresponding labels, model paths and weights,
# this function will return a prediction matrix, containing a prediction on each 
# example for each model. It will also return names of the models.
def load_models_and_predict(X_chunk, y_chunk, model_paths, h5_paths):

    # Construct a matrix that holds a prediction value for each image and each model in the ensemble
    prediction_matrix = np.zeros((X_chunk.shape[0], len(model_paths)))
    model_names       = list()      # Keep track of model names
    individual_scores = list()      # Keep track of model acc on an individual basis
    
    # Load each model and perform prediction with it.
    for model_idx, model_path in enumerate(model_paths):

        # 1.0 - Set model parameters
        yaml_path = glob.glob(os.path.join(model_path, "run.yaml"))[0]              
        settings_yaml = load_settings_yaml(yaml_path, verbatim=False)                           # Returns a dictionary object.
        params = Parameters(settings_yaml, yaml_path, mode="no_training")                       # Create Parameter object
        params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.
        
        # 2.0 - Create a dataGenerator object, because the network class wants it
        dg = DataGenerator(params, mode="no_training", do_shuffle_data=False, do_load_validation=False)

        # 3.0 - Construct a Network object that has a model as property.
        network = Network(params, dg, training=False, verbatim=False)
        network.model.load_weights(h5_paths[model_idx])

        # Keep track of model name
        model_names.append(network.params.model_name)

        # 4.0 - Evaluate on validation chunk
        if "resnet_single_newtr_last_last_weights_only" in model_path:
            X_chunk = dstack_data(X_chunk)
        evals = network.model.evaluate(X_chunk, y_chunk, verbose=0)
        print("\n\nIndividual Evaluations:")
        print("Model name: {}".format(network.params.model_name))
        for met_idx in range(len(evals)):
            print("{} = {}".format(network.model.metrics_names[met_idx], evals[met_idx]))
            individual_scores.append((network.model.metrics_names[met_idx], evals[met_idx]))
        print("")

        # 5.0 - Predict on validation chunk
        predictions = network.model.predict(X_chunk)
        prediction_matrix[:,model_idx] = np.squeeze(predictions)

    # Networks are also need later on, therefore we need to add them to a list
    return prediction_matrix, model_names, individual_scores


# Given a prediction matrix it will try to find weights for each model.
# Via a minimization problem.
def find_ensemble_weights(prediction_matrix, y_chunk, model_names, method="Nelder-Mead", verbatim=False):
    
    # Randomly initialize the initial weight vector
    w0 = np.random.random_sample((len(model_names),))

    # We want to pass arguments to the minimization function. This is done with a partial function.
    f_partial = partial(mean_square_error, y_chunk, prediction_matrix)

    # Perform the minimization with the scipy package
    res = minimize(f_partial, w0, method='Nelder-Mead', tol=1e-6, options={'disp' : True})

    # Collect model weights and print to user. (Softmax the weights so that they sum to 1.)
    model_weights = res.x
    model_weights = softmax(model_weights)
    if verbatim:
        print("Model weights\t = {}".format(model_weights))
        print("Weights Sum\t = {}".format(np.sum(model_weights)))
        print("Model names\t = {}".format(model_names))
        print("Model weights data type\t = {}".format(type(model_weights)))
    return model_weights


# Evaluating an ensemble is done by performing a dot product of the prediction matrix
# with the model weight vector. Followed by determining error, and returning accuracy.
def evaluate_ensemble(prediction_matrix, y, model_weights, threshold=0.5):
    print("\n\n===================\nEnsemble Evaluation:")

    # In order to get ensemble prediction we need to do a dot product
    # of prediction vector with model weight vector.
    y_hat = np.dot(prediction_matrix, model_weights)

    y_hat_copy = np.copy(y_hat)
    # Count TP, TN, FP, FN
    y_hat[y_hat < threshold] = 0.0
    y_hat[y_hat >= threshold] = 1.0

    # Count mistakes
    error_count = np.sum(np.absolute(y - y_hat))
    print("mistakes count: {} out of {}".format(error_count, y_hat.shape[0]))

    # Calculate accuracy
    error_percentage = error_count/y_hat.shape[0]
    acc = 1.0 - error_percentage
    return acc, y_hat_copy


# Loading a chunk into memory
def load_chunk_train(chunksize, X_lenses, X_negatives, X_sources, data_type, mock_lens_alpha_scaling):
    start_time = time.time()

    # Half a chunk positive and half negative
    num_positive = int(chunksize / 2)
    num_negative = int(chunksize / 2)
    
    # Get mock lenses data and labels
    X_pos, y_pos = merge_lenses_and_sources(X_lenses, X_sources, num_positive, data_type, mock_lens_alpha_scaling, do_deterministic=False)
        
    # Store Negative data in numpy array and create label vector
    X_neg = np.empty((num_negative, X_pos.shape[1], X_pos.shape[2], X_pos.shape[3]), dtype=data_type)
    y_neg = np.zeros(X_neg.shape[0], dtype=data_type)

    # We want a 80% probability of selecting from the contaminants set, and 20% probability of selecting an LRG from the lenses set.
    negative_sample_contaminant_prob = 0.8
    for i in range(num_negative):
        if random.random() <= negative_sample_contaminant_prob:
            X_neg[i] = np.sqrt(X_negatives[random.randint(0, X_negatives.shape[0] - 1)])        #sqrt stretch is needed, because the positive examples are also sqrt stretched
        else:
            X_neg[i] = np.sqrt(X_lenses[random.randint(0, X_lenses.shape[0] - 1)])        #sqrt stretch is needed, because the positive examples are also sqrt stretched

    # Concatenate the positive and negative examples into one chunk (also the labels)
    X_chunk = np.concatenate((X_pos, X_neg))
    y_chunk = np.concatenate((y_pos, y_neg))

    print("Creating training chunk took: {}, chunksize: {}".format(hms(time.time() - start_time), chunksize), flush=True)
    return X_chunk, y_chunk


# Construct a multi-layer-perceptron
# takes as input the predictions of the ensemble and outputs the prediction made by the mlp
def get_mlp(input_size=5, output_size=1, name='ensemble_mlp'):

    # Define Architecture
    inputs  = Input(shape=(input_size,))
    dense   = Dense(10, activation='relu')(inputs)
    dense   = Dense(15, activation='relu')(dense)
    dense   = Dense(10, activation='relu')(dense)
    outputs = Dense(output_size, activation='sigmoid')(dense)

    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    #print model architecture
    print("Ensemble MLP architecture")
    model.summary()

    #compile the model with optimizer, loss and metric
    model.compile(optimizer=optimizers.Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


# commment
def train_mlp(mlp, prediction_matrix, y_chunk, epochs=10, x_val=None, y_val=None):
    history = mlp.fit(prediction_matrix, y_chunk, batch_size=64, epochs=epochs, validation_data=(x_val, y_val))
    return mlp


    
############################################################ Script ############################################################

def main():

    # 0.0 - Deal with input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_name", help="Name of the ensemble. There will be a folder with the given name.", default="test_ensemble", required=False)
    parser.add_argument("--sample_size", help="The amount of images used in the validation set to optimize ensemble model weights. A maximum of 551 can be used.", default=1024, required=False)
    parser.add_argument("--method", help="What ensemble method should be used? To determine model weights? For example: Nelder-Mead.", default="Nelder-Mead", required=False)
    args = parser.parse_args()

    ### set root directory
    root_dir_ensembles = "ensembles"

    # 0.1 - Create folder that holds Ensembles
    create_dir_if_not_exists(root_dir_ensembles, verbatim=False)

    # 1.0 - Fix memory leaks if running on tensorflow 2
    tf.compat.v1.disable_eager_execution()

    # 2.0 - Model Selection from directory - Select multiple models
    model_paths = choose_ensemble_members()

    # 2.5 - Define Ensemble name and create a folder for it
    name = args.ensemble_name + "_samples_" + str(args.sample_size)
    ensemble_dir = os.path.join(root_dir_ensembles, name)
    create_dir_if_not_exists(ensemble_dir)

    # 3.0 - Select a weights file. There are 2 for each model. Selected based on either validation loss or validation metric. The metric can differ per model.
    h5_paths = get_h5_path_dialog(model_paths)

    # 4.0 - Select random sample from the data (with replacement)
    sample_size = int(args.sample_size)
    # sample_size = int(input("How many samples do you want to create and run (int): "))
    # if args.method == "mlp":
    sources_fnames_train, lenses_fnames_train, negatives_fnames_train = get_samples(size=sample_size, type_data="train", deterministic=False)
    sources_fnames_val, lenses_fnames_val, negatives_fnames_val = get_samples(size=551, type_data="validation", deterministic=False)
    sources_fnames_test, lenses_fnames_test, negatives_fnames_test = get_samples(size=551, type_data="test", deterministic=True)
    # if args.method == "Nelder-Mead":
    #     sources_fnames, lenses_fnames, negatives_fnames = get_samples(size=sample_size, type_data="train", deterministic=False)

    # 5.0 - Load unnormalized data in order to calculate the amount of noise in a lens.
    # Create a chunk of data that each neural network understands (preprocessed quite identically)
    PSF_r = compute_PSF_r()  # Used for sources only
    lenses_train      = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames_train)
    sources_train     = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames_train)
    negatives_train   = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames_train)
    
    lenses_val       = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames_val)
    sources_val      = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames_val)
    negatives_val    = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames_val)

    lenses_test      = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames_test)
    sources_test     = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames_test)
    negatives_test   = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames_test)

    # lets hook in at this point, in order to train an mlp instead.
    mlp = get_mlp(input_size=len(model_paths), output_size=1, name=str(args.ensemble_name))

    num_chunks = 2
    for i in range(num_chunks):

        # 6.0 - Load a 50/50 positive/negative chunk into memory
        # if args.method == "Nelder-Mead":
        X_chunk_val, y_chunk_val = load_chunk_val(lenses_val, sources_val, negatives_val, mock_lens_alpha_scaling=(0.02, 0.30))
        # if args.method == "mlp":
        X_chunk, y_chunk = load_chunk_train(int(args.sample_size), lenses_train, negatives_train, sources_train, np.float32, mock_lens_alpha_scaling=(0.02, 0.30))

        # chunk of code that can be used to visualize some images of the given set:
        if False:
            for i in range(X_chunk.shape[0]):
                img = np.squeeze(X_chunk[i])
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                print(y_chunk)
                plt.show()

        # 7.0 - Load ensemble members and perform prediction with it.
        prediction_matrix, model_names, individual_scores = load_models_and_predict(X_chunk, y_chunk, model_paths, h5_paths)

        prediction_matrix_val, model_names_val, individual_scores_val = load_models_and_predict(X_chunk_val, y_chunk_val, model_paths, h5_paths)

        mlp = train_mlp(mlp, prediction_matrix, y_chunk, epochs=5, x_val=prediction_matrix_val, y_val=y_chunk_val)

    # Create a test chunk:
    X_chunk_test, y_chunk_test = load_chunk_test(np.float32, (0.02, 0.30), lenses_test, sources_test, negatives_test)

    # if we created an mlp to predict y_hat based on all ensemble members their predictions:
    if args.method == "mlp":
        mlp.save_weights(os.path.join(ensemble_dir, "ensemble_mlp.h5"))
        prediction_matrix_test, model_names, individual_scores = load_models_and_predict(X_chunk_test, y_chunk_test, model_paths, h5_paths)
        y_hat = mlp.predict(prediction_matrix_test)
        print("y_hat mlp ensemble: {}".format([y for y in np.squeeze(y_hat)]))
        loss, acc = mlp.evaluate(prediction_matrix_test, y_chunk_test)
        print("mlp loss = {}, acc = {}".format(loss, acc))
    
    # 9.0 - Ensemble Evaluation on Nelder-Mead
    threshold = 0.5
    model_weights = find_ensemble_weights(prediction_matrix, y_chunk, model_names, method=args.method, verbatim=True)
    if args.method == "Nelder-Mead":
        acc, y_hat = evaluate_ensemble(prediction_matrix_val, y_chunk_val, model_weights, threshold=threshold)
    print("Model Weight Vector:\t{}".format(model_weights))
    print("\n\nAccuracy of Ensemble\t{:.3f} at threshold: {}".format(acc, threshold))
    print("Model names:\t\t{}".format(model_names))

    ### f_beta graph and its paramters
    beta_squarred           = 0.03                                  # For f-beta calculation
    stepsize                = 0.01                                  # For f-beta calculation
    threshold_range         = np.arange(stepsize, 1.0, stepsize)    # For f-beta calculation
    colors = ['r', 'c', 'green', 'orange', 'lawngreen', 'b', 'plum', 'darkturquoise', 'm']

    # I would like to see an f_beta figure of the ensemble that can be compared with a single model's f_beta figure.
    f_betas, precision_data, recall_data = [], [], []
    for p_threshold in threshold_range:
        (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(y_hat, y_chunk_val, p_threshold, beta_squarred)
        f_betas.append(F_beta)
        precision_data.append(precision)
        recall_data.append(recall)

    # step 7.2 - Plotting all lines
    plt.plot(list(threshold_range), f_betas, colors[0], label = "f_beta")
    plt.plot(list(threshold_range), precision_data, ":", color=colors[0], alpha=0.9, linewidth=3, label="Precision")
    plt.plot(list(threshold_range), recall_data, "--", color=colors[0], alpha=0.9, linewidth=3, label="Recall")
    
    # set up plot aesthetics
    plt.xlabel("p threshold")
    plt.ylabel("F")
    plt.title("Ensemble F_beta, where Beta = {0:.2f}".format(math.sqrt(beta_squarred)))
    figure = plt.gcf() # get current figure
    axes = plt.gca()
    # axes.set_ylim([0.63, 1.00])
    figure.set_size_inches(12, 8)       # (12,8), seems quite fine
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.savefig('{}.png'.format(os.path.join(ensemble_dir, "f_beta_plot")))
    plt.show()

    # 10.0 - Create a dictionary that will hold Ensemble parameters
    acc = float("{:.03f}".format(acc))   # this conversion was somehow necessary.
    precision = precision_data[(len(threshold_range)//2) + 1]
    recall    = recall_data[(len(threshold_range)//2) + 1]
    fbeta     = f_betas[(len(threshold_range)//2) + 1]
    _write_params_to_yaml(model_names, model_weights, args, ensemble_dir, acc, threshold, sample_size, precision, recall, fbeta)

    

if __name__ == "__main__":
    main()