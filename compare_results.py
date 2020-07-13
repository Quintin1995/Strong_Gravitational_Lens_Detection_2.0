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


def compare_plot_models(comparing_headerName_df, dfs, jsons, json_comp_key, do_legend):
    for idx in range(len(dfs)):         # loop over each model dataframe
        data = dfs[idx][comparing_headerName_df]
        
        if comparing_headerName_df == "time":
            data = list(data)
            formatted_time_data = []     # in minutes
            for timestamp in data:
                parts = timestamp.split(":")
                formatted_time_data.append(int(parts[0])*60 + int(parts[1]) + int(parts[2])/60)
            data = formatted_time_data

        plt.plot(data, label = str(json_comp_key) + ": " + str(jsons[idx][json_comp_key]))

    plt.title(comparing_headerName_df)
    plt.ylabel(comparing_headerName_df)
    plt.xlabel("Trained Chunks")
    if do_legend:
        plt.legend()
    plt.show()

####################################
root_models = "models"

######### Settable Paramters
models = [
    "07_04_2020_18h_27m_45s_test_ram_logging_ownPC",
    "07_04_2020_20h_03m_34s_test_ram_logging_peregrine"
]
comparing_headerName_df = "ram_usage"
json_comp_key           = "fraction_to_load_lenses"
do_legend               = False
#########

## 1.0 - Get list dataframes
dfs, csv_paths = get_dataframes(models)

## 2.0 - Get list of jsons
jsons, json_paths = get_jsons(models)

## 3.0 - Plot the data from the csvs - legend determined by json parameter dump file
compare_plot_models(comparing_headerName_df, dfs, jsons, json_comp_key, do_legend)

## 4.0 - Calculate f-beta score per model - based on a couple of validation chunks
