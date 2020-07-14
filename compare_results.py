import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
import json
from utils import *
from Parameters import *
from Parameters import Parameters
from DataGenerator import *
from Network import *
import csv
import math


# Count the number of true positive, true negative, false positive and false negative in for a prediction vector relative to the label vector.
def count_TP_TN_FP_FN_and_FB(prediction_vector, y_test, threshold, beta_squarred, verbatim = False):
    TP = 0 #true positive
    TN = 0 #true negative
    FP = 0 #false positive
    FN = 0 #false negative

    for idx, pred in enumerate(prediction_vector):
        if pred >= threshold and y_test[idx] >= threshold:
            TP += 1
        if pred < threshold and y_test[idx] < threshold:
            TN += 1
        if pred >= threshold and y_test[idx] < threshold:
            FP += 1
        if pred < threshold and y_test[idx] >= threshold:
            FN += 1

    tot_count = TP + TN + FP + FN
    
    precision = TP/(TP + FP) if TP + FP != 0 else 0
    recall    = TP/(TP + FN) if TP + FN != 0 else 0
    fp_rate   = FP/(FP + TN) if FP + TN != 0 else 0
    accuracy  = (TP + TN) / len(prediction_vector) if len(prediction_vector) != 0 else 0
    F_beta    = (1+beta_squarred) * ((precision * recall) / ((beta_squarred * precision) + recall)) if ((beta_squarred * precision) + recall) else 0
    
    if verbatim:
        if tot_count != len(prediction_vector):
            print("Total count {} of (TP, TN, FP, FN) is not equal to the length of the prediction vector: {}".format(tot_count, len(prediction_vector)), flush=True)

        print("Total Count {}\n\tTP: {}, TN: {}, FP: {}, FN: {}".format(tot_count, TP, TN, FP, FN), flush=True)
        print("precision = {}".format(precision), flush=True)
        print("recall    = {}".format(recall), flush=True)
        print("fp_rate   = {}".format(fp_rate), flush=True)
        print("accuracy  = {}".format(accuracy), flush=True)
        print("F beta    = {}".format(F_beta), flush=True)

    return TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta


# Construct dataframes of csv files in the given model folders. 
# This will be used for plotting data about the model
# csv - Output Dump
def get_dataframes(models):
    print("\n\nReadin in csv data:")
    dfs = []    # list of dataframes - for each csv
    model_paths_csv = []
    for idx, model in enumerate(models):
        print("Model: {} - csv data".format(idx))
        path = os.path.join(root_models, model)
        history_csv_path = glob.glob(path + "/*history.csv")[0]
        model_paths_csv.append(history_csv_path)
        dfs.append(pd.read_csv(history_csv_path))
        # dfs.insert(0, pd.read_csv(history_csv_path))
        print("path = {}".format(history_csv_path))
        print(dfs[idx].head())
        print("\n")
    # dfs.reverse()
    # model_paths_csv.reverse()
    return dfs, model_paths_csv


# Construct jsons of josn files in the given model folders.
# This will be used to get model paramters when plotting
# json - Input Parameter Dump
def get_jsons(models):
    print("\n\njson files: ")
    jsons = []
    model_paths_json = []
    for idx, model in enumerate(models):
        print("Model: {} - json data".format(idx))
        path = os.path.join(root_models, model)
        paramDump_json_path = glob.glob(path + "/*.json")[0]
        model_paths_json.append(paramDump_json_path)
        jsons.append(json.load(open(paramDump_json_path)))
        # jsons.insert(0, json.load(open(paramDump_json_path)))
        print("path = {}".format(paramDump_json_path))
        for i in jsons[idx]:
            print("\t" + i + ": " + str(jsons[idx][i]))
        print("\n")
    # jsons.reverse()
    # model_paths_json.reverse()
    return jsons, model_paths_json


# For each model path in models find a .h5 file path and return it.
def get_h5s_paths(models):
    print("\n\nh5 files: ")
    paths_h5s = []
    for idx, model in enumerate(models):
        print("Model: {} - h5 file".format(idx))
        path = os.path.join(root_models, model)
        h5_path = glob.glob(path + "/*.h5")[0]
        paths_h5s.append(h5_path)
        # paths_h5s.insert(0, h5_path)
    # paths_h5s.reverse()
    return paths_h5s


# dstack the data to three channels instead of one
def dstack_data(data):
    dstack_data = np.empty((data.shape[0], data.shape[1], data.shape[2], 3), dtype=np.float32)
    for i in range(data.shape[0]):
        img = data[i]
        dstack_data[i] = np.dstack((img,img,img))
    return dstack_data


def average_prediction_results(network, data, avg_iter_counter=10, verbose=False):
    avg_preds = np.zeros((data.shape[0],1), dtype=np.float32)
    for i in range(avg_iter_counter):
        if verbose:
            print("avg iter counter = {}".format(i))
        prediction_vector = network.model.predict(data)
        avg_preds = np.add(avg_preds, prediction_vector)
    avg_preds = avg_preds / avg_iter_counter
    if verbose:
        print("Length prediction vector: {}".format(len(avg_preds)), flush=True)
    return avg_preds


# Load validation chunk and calculate per model folder its model performance evaluated on f-beta score
def store_fbeta_results(models, jsons, json_comp_key):
    for idx, model_folder in enumerate(models):
        
        # Load settings of the model
        yaml_path = glob.glob(os.path.join(root_models, model_folder) + "/*.yaml")[0]
        settings_yaml = load_settings_yaml(yaml_path)
        
        # Set Parameters - and overload fraction to load sources - because not all are needed and it will just slow things down for now.
        params = Parameters(settings_yaml, yaml_path, mode="no_training")
        params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.
        if params.model_name == "Baseline_Enrico":
            params.img_dims = (101,101,3)
        
        # Construct a neural network with the same architecture as that it was trained with.
        resnet18 = Network(params.net_name, params.net_learning_rate, params.net_model_metrics, params.img_dims, params.net_num_outputs, params)
        resnet18.model.trainable = False
        
        # Load weights of the neural network
        resnet18.model.load_weights(paths_h5s[idx])

        # Define a DataGenerator that can generate validation chunks based on validation data.
        dg = DataGenerator(params, mode="no_training", do_shuffle_data=False)     #do not shuffle the data in the data generator
        placeholder = 1000   #this num doesn't matter - its chunksize. But for this validation chunk it is being overriden in the load_chunk function, by the do_deterministic boolean
        
        X_validation_chunk, y_validation_chunk = dg.load_chunk(placeholder, dg.Xlenses_validation, dg.Xnegatives_validation, dg.Xsources_validation, params.data_type, params.mock_lens_alpha_scaling, do_deterministic=True)
        
        # dstack images for enrico neural network
        if params.model_name == "Baseline_Enrico":
            X_validation_chunk = dstack_data(X_validation_chunk)

        # Predict the labels of the validation chunk on the loaded neural network - averaged over 'avg_iter_counter' predictions
        avg_preds = average_prediction_results(resnet18, X_validation_chunk, avg_iter_counter=10, verbose=False)
        
        # Define paths to filenames for f-beta saving and its plot
        f_beta_full_path = os.path.join(root_models, model_folder, "f_beta_results.csv")
        full_path_fBeta_figure = os.path.join(root_models, model_folder, "f_beta_graph.png")

        # Begin f-beta calculation and store into csv file
        f_betas = []
        with open(f_beta_full_path, 'w', newline='') as f_beta_file:
            writer = csv.writer(f_beta_file)
            writer.writerow(["p_threshold", "TP", "TN", "FP", "FN", "precision", "recall", "fp_rate", "accuracy", "f_beta"])
            for p_threshold in threshold_range:
                (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(avg_preds, y_validation_chunk, p_threshold, beta_squarred)
                f_betas.append(F_beta)
                writer.writerow([str(p_threshold), str(TP), str(TN), str(FP), str(FN), str(precision), str(recall), str(fp_rate), str(accuracy), str(F_beta)])
        print("saved csv with f_beta scores to: ".format(f_beta_full_path), flush=True)

        if label_override:
            plt.plot(list(threshold_range), f_betas, label = models[idx])
        else:
            plt.plot(list(threshold_range), f_betas, label = str(json_comp_key) + ": " + str(jsons[idx][json_comp_key]))
        plt.xlabel("p threshold")
        plt.ylabel("F")
        plt.title("F_beta score - Beta = {0:.2f}".format(math.sqrt(beta_squarred)))

        plt.savefig(full_path_fBeta_figure)
        print("figure saved: {}".format(full_path_fBeta_figure), flush=True)
    if do_legend:
        plt.legend()
    plt.show()
    

def compare_plot_models(comparing_headerName_df, dfs, jsons, json_comp_key, do_legend):
    for idx in range(len(dfs)):         # loop over each model dataframe
        data = dfs[idx][comparing_headerName_df]
        
        # time needs to be formatted before it can be plotted
        if comparing_headerName_df == "time":
            data = list(data)
            formatted_time_data = []     # in minutes
            for timestamp in data:
                parts = timestamp.split(":")
                formatted_time_data.append(int(parts[0])*60 + int(parts[1]) + int(parts[2])/60)
            data = formatted_time_data

        if label_override:
            plt.plot(data, label = models[idx])
        else:
            plt.plot(data, label = str(json_comp_key) + ": " + str(jsons[idx][json_comp_key]))
            print(str(str(json_comp_key) + ": " + str(jsons[idx][json_comp_key])))

    plt.title(comparing_headerName_df)
    plt.ylabel(comparing_headerName_df)
    plt.xlabel("Trained Chunks")
    if do_legend:
        plt.legend()
    plt.show()

############## Parameters ##############
root_models = "models"

######### Settable Paramters
models = [
    # "resnet_single_newtr_last_last_weights_only",
    "07_11_2020_14h_39m_01s_test_ram_logging_3700chunks_ownPC",
    "07_11_2020_15h_11m_03s_test_ram_logging_pere4"
]
json_comp_key           = "model_name"              # is the label in generated plots
do_legend               = True                      # whether legend needs to be present in the plot
label_override          = False                     # assign labels in the legend based on model folder instead of json_comp_key
show_all_step4_plots    = False

# f-beta
beta_squarred = 0.03                                    # For f-beta calculation
stepsize = 0.01                                         # For f-beta calculation
threshold_range = np.arange(stepsize,1.0,stepsize)      # For f-beta calculation
#########

## 1.0 - Get list dataframes
if show_all_step4_plots:
    dfs, csv_paths = get_dataframes(models)

## 2.0 - Get list of jsons
jsons, json_paths = get_jsons(models)

## 3.0 - get list of .h5 files
paths_h5s = get_h5s_paths(models)

## 4.0 - Plot the data from the csvs - legend determined by json parameter dump file
if show_all_step4_plots:
    for columnname in dfs[0].columns:
        if columnname == "chunk":
            continue
        compare_plot_models(columnname, dfs, jsons, json_comp_key, do_legend)

## 5.0 - Calculate f-beta score per model - based on validation data
if True:
    store_fbeta_results(models, jsons, json_comp_key)