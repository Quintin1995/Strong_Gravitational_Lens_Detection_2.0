import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
import json


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
        history_csv_path = glob.glob(path + "/*.csv")[0]
        model_paths_csv.append(history_csv_path)
        dfs.append(pd.read_csv(history_csv_path))
        print("path = {}".format(history_csv_path))
        print(dfs[idx].head())
        print("\n")
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
        print("path = {}".format(paramDump_json_path))
        for i in jsons[idx]:
            print("\t" + i + ": " + str(jsons[idx][i]))
        print("\n")
    return jsons, model_paths_json


def compare_plot_models(comparing_headerName_df, dfs, jsons, json_comp_key):
    axiss = []
    for idx in range(len(dfs)):
        data = dfs[idx][comparing_headerName_df]
        plt.plot(data, label = str(json_comp_key) + ": " + str(jsons[idx][json_comp_key]))


    plt.title(comparing_headerName_df)
    plt.ylabel(comparing_headerName_df)
    plt.xlabel("Trained Chunks")
    plt.legend()
    plt.show()

####################################
root_models = "models"

######### Settable Paramters
models = [
    "07_03_2020_19h_41m_46s_test1_ja",
    "07_03_2020_16h_25m_43s_yes_avg_pool_multi_ja"
]
comparing_headerName_df = "loss"
json_comp_key           = "fraction_to_load_lenses"
#########

## 1.0 - Get list dataframes
dfs, csv_paths = get_dataframes(models)

## 2.0 - Get list of jsons
jsons, json_paths = get_jsons(models)

## 3.0 - Plot the data
compare_plot_models(comparing_headerName_df, dfs, jsons, json_comp_key)