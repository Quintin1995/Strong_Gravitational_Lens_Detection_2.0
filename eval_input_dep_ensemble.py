import os
import glob
import functools
import matplotlib.pyplot as plt
import numpy as np
import math
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import RMSprop, Adam
from scipy.special import softmax

from utils import load_settings_yaml, get_fnames_from_disk, compute_PSF_r, count_TP_TN_FP_FN_and_FB
from create_input_dep_ensemble import get_simple_ensemble_model, get_resnet18_ensemble_model, load_networks, load_and_normalize_img, load_chunk_test

#### DESCRIPTION ###
# Given a folder containing multiple instances of an ensemble run.
# This script will evaluate each ensemble and average the results
# in the form of an f_beta graph.
## Most code is copied from create_input_dep_ensemble


# Directory stuff
rootdir       = "ensembles"
subdir        = "final_ensemble_experiment"
ensembles_dir = os.path.join(rootdir, subdir)
ens_dirs      = os.listdir(ensembles_dir)
colors = ['r', 'c', 'green', 'orange', 'lawngreen', 'b', 'plum', 'darkturquoise', 'm']


# Load test data into RAM before evaluation of each ensemble
if True:
    sources_fnames_test, lenses_fnames_test, negatives_fnames_test  = get_fnames_from_disk(1.0, 1.0, 1.0, type_data="test", deterministic=False)
    PSF_r = compute_PSF_r()  # Used for sources only
    with Pool(24) as p:
        lenses_test_f     = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        lenses_test       = np.asarray(p.map(lenses_test_f, enumerate(lenses_fnames_test), chunksize=128), np.float32)

        sources_test_f    = functools.partial(load_and_normalize_img, np.float32, True, "per_image", PSF_r)
        sources_test      = np.asarray(p.map(sources_test_f, enumerate(sources_fnames_test), chunksize=128), np.float32)

        negatives_test_f  = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        negatives_test    = np.asarray(p.map(negatives_test_f, enumerate(negatives_fnames_test), chunksize=128), np.float32)

    # Use the test data for {accuracy, f_beta, precision, recall} performance metric evaluations.
    X_chunk_test, y_chunk_test   = load_chunk_test(np.float32, (0.02, 0.30), lenses_test, sources_test, negatives_test)


f_beta_vectors = list()
precision_data_vectors = list()
recall_data_vectors = list()

# Loop over each ensemble
for ens_dir in ens_dirs:
    
    # Get settings of ensemble
    run_file_path = glob.glob(os.path.join(ensembles_dir, ens_dir, "run*.yaml"))[0]
    settings_dict = load_settings_yaml(run_file_path)

    # Fix memory leaks if running on tensorflow 2
    tf.compat.v1.disable_eager_execution()

    # Model Selection from directory - Select multiple models
    model_paths = list(settings_dict["models"])

    # Construct input dependent ensemble model
    if settings_dict["network_name"] == "simple_net":
        ens_model = get_simple_ensemble_model(settings_dict, input_shape=(101,101,1), num_outputs=len(model_paths))
    if settings_dict["network_name"] == "resnet18":
        ens_model = get_resnet18_ensemble_model(settings_dict, input_shape=(101,101,1), num_outputs=len(model_paths))

    # Load the individual networks/models into a list and keep track of their names.
    networks, model_names = load_networks(model_paths)

    # Load the weight of the ensemble model
    ens_model.load_weights(os.path.join(ensembles_dir, ens_dir, "ensemble_model_val_cat_acc.h5"))

    # Perform prediction on the test set for the ensemble model
    print("\n\n ensemble model predict:")
    ens_preds = ens_model.predict(X_chunk_test)
    for idx, pred in enumerate(ens_preds):
        if idx < 10:
            print(pred)

    # Loop over individual models and predict
    print("\n\n members prediction:")
    individual_predictions = np.zeros((ens_preds.shape))
    for net_idx, network in enumerate(networks):
        individual_predictions[:,net_idx] = np.squeeze(network.model.predict(X_chunk_test))
    for idx, pred in enumerate(individual_predictions):
        if idx < 10:
            print(individual_predictions[idx])

    # Now we should use the prediction of the ensemble model as weights for the individual predictions of its members.
    print("\n\n final prediction:")
    ens_y_hat = np.zeros(y_chunk_test.shape)
    for i in range(individual_predictions.shape[0]):
        ens_weights     = ens_preds[i]
        members_row     = individual_predictions[i]
        ens_y_hat[i]    = np.dot(ens_weights, members_row)
        if i < 10:
            print(ens_y_hat[i])

    #### start f_beta plot of current ensemble
    ### f_beta graph and its paramters
    beta_squarred           = 0.03                                  # For f-beta calculation
    stepsize                = 0.01                                  # For f-beta calculation
    threshold_range         = np.arange(stepsize, 1.0, stepsize)    # For f-beta calculation

    # I would like to see an f_beta figure of the ensemble that can be compared with a single model's f_beta figure.
    f_betas, precision_data, recall_data = [], [], []
    print("\n\n accuracies per f_beta bin:")
    for p_threshold in threshold_range:
        (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(ens_y_hat, y_chunk_test, p_threshold, beta_squarred)
        print("acc: {0:.3f} on threshold: {1:.3f}".format(accuracy, p_threshold))
        f_betas.append(F_beta)
        precision_data.append(precision)
        recall_data.append(recall)
    # Keep track of each fbeta, precision and recall curve for each ensemble:
    f_beta_vectors.append(f_betas)
    precision_data_vectors.append(precision_data)
    recall_data_vectors.append(recall_data)

# step 7.0 - calculate std and mean - based on f_beta_vectors
colls = list(zip(*f_beta_vectors))
means = np.asarray(list(map(np.mean, map(np.asarray, colls))))
stds = np.asarray(list(map(np.std, map(np.asarray, colls))))

# step 7.1 - define upper and lower limits
upline  = np.add(means, stds)
lowline = np.subtract(means, stds)

# Step 7.1.1 - Calculate mean precision and recall rates
colls_pre = list(zip(*precision_data_vectors))
precision_mu = np.asarray(list(map(np.mean,(map(np.asarray, colls_pre)))))
colls_recall = list(zip(*recall_data_vectors))
recall_mu = np.asarray(list(map(np.mean,(map(np.asarray, colls_recall)))))

# step 7.2 - Plotting all lines
plt.plot(list(threshold_range), precision_mu, ":", color=colors[0], label="precision mean", alpha=0.9, linewidth=3)
plt.plot(list(threshold_range), recall_mu, "--", color=colors[0], label="recall mean", alpha=0.9, linewidth=3)
plt.plot(list(threshold_range), upline, colors[0])
plt.plot(list(threshold_range), means, colors[0], label="3 Members Ensemble")
plt.plot(list(threshold_range), lowline, colors[0])
plt.fill_between(list(threshold_range), upline, lowline, color=colors[0], alpha=0.5) 

plt.xlabel("p threshold")
plt.ylabel("F")
plt.title("F_beta score - Beta = {0:.2f}".format(math.sqrt(beta_squarred)))
figure = plt.gcf() # get current figure
axes = plt.gca()
axes.set_ylim([0.63, 1.00])
figure.set_size_inches(12, 8)       # (12,8), seems quite fine
fname = '{}.png'.format(os.path.join(ensembles_dir, ens_dir, "f_beta_plot"))
plt.savefig(fname, dpi=100)
plt.grid(color='grey', linestyle='dashed', linewidth=1)
plt.legend()
plt.show()