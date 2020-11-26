import os
import glob
from utils import load_settings_yaml, count_TP_TN_FP_FN_and_FB
from Parameters import *
import numpy as np
from DataGenerator import *
from Network import *


########################################## Functions ##########################################




########################################## Params ##########################################
names               = ["binary_crossentropy", "f_beta"]
metric_interests    = ["loss", "metric"]
do_eval             = True
fraction_to_load_sources_test = 0.5

### f_beta graph and its paramters
beta_squarred           = 0.03                                  # For f-beta calculation
stepsize                = 0.01                                  # For f-beta calculation
threshold_range         = np.arange(stepsize, 1.0, stepsize)    # For f-beta calculation


root = os.path.join("models", "test_exp")
########################################## Script ##########################################
# pre-checks
assert len(names) == len(metric_interests)


# for each member in the experiment keep a list of models that are of the same type and hyper-params
model_collections = list()

# create collections of models - models are grouped in lists
for name in names:
    model_list = glob.glob(root + "/*{}".format(name))
    model_collections.append(model_list)


for collection_idx, model_collection in enumerate(model_collections):

    # Keep track of results per model
    f_beta_vectors = []
    precision_data_vectors = []
    recall_data_vectors = []

    # loop over each model - 1. Initialize, 2. Predict, 3. Store results
    for model_folder in model_collection:

        # Step 1.0 - Load settings of the model
        yaml_path = glob.glob(os.path.join(model_folder) + "/*.yaml")[0]
        settings_yaml = load_settings_yaml(yaml_path)

        # Step 2.0 - Set Parameters - and overload fraction to load sources - because not all are needed and it will just slow things down for now.
        params = Parameters(settings_yaml, yaml_path, mode="no_training")
        params.fraction_to_load_sources_test = fraction_to_load_sources_test

        params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.
        if params.model_name == "Baseline_Enrico":
            params.img_dims = (101,101,3)

        # Step 3.0 - Define a DataGenerator that can generate test chunks based on test data.
        dg = DataGenerator(params, mode="no_training", do_shuffle_data=False, do_load_test_data=True, do_load_validation=False)     #do not shuffle the data in the data generator
        
        # Step 4.0 - Construct a neural network with the same architecture as that it was trained with.
        network = Network(params, dg, training=False) # The network needs to know hyper-paramters from params, and needs to know how to generate data with a datagenerator object.
        network.model.trainable = False

        # Step 5.0 - Load weights of the neural network
        h5_path = glob.glob(model_folder + "/checkpoints/*{}.h5".format(metric_interests[collection_idx]))[0]
        network.model.load_weights(h5_path)

        # Step 6.0 - Load the data
        X_test_chunk, y_test_chunk = dg.load_chunk_test(params.data_type, params.mock_lens_alpha_scaling)

        # Step 6.1 - dstack images for enrico neural network
        if params.model_name == "Baseline_Enrico":
            X_test_chunk = dstack_data(X_test_chunk)

        # Step 6.2 - Predict the labels of the test chunk on the loaded neural network - averaged over 'avg_iter_counter' predictions
        preds = network.model.predict(X_test_chunk)

        # Step 6.3 - Also calculate an evaluation based on the models evaluation metric
        if do_eval:
            results = network.model.evaluate(X_test_chunk, y_test_chunk, verbose=0)
            print("\n\n" + model_folder)
            for met_idx in range(len(results)):
                print("Test {} = {}".format(network.model.metrics_names[met_idx], results[met_idx]))

        # Step 6.4 - Begin f-beta calculation
        f_betas = []
        precision_data = []
        recall_data = []
        for p_threshold in threshold_range:
            (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(preds, y_test_chunk, p_threshold, beta_squarred)
            f_betas.append(F_beta)
            precision_data.append(precision)
            recall_data.append(recall)
        f_beta_vectors.append(f_betas)
        precision_data_vectors.append(precision_data)
        recall_data_vectors.append(recall_data)

        f_beta_matrix = np.asarray(f_beta_vectors)
        mu_fbeta = np.mean(f_beta_matrix, axis=0)
        std_fbeta = np.std(f_beta_matrix, axis=0)

        