import os
from compare_results import set_experiment_folder, set_models_folders


# Opens dialog with the user to select a folder that contains models.
def get_model_path(root_dir="models"):

    #Select which experiment path to take in directory structure
    experiment_folder = set_experiment_folder(root_folder=root_dir)

    # Can select 1 or multiple models.
    models_paths = set_models_folders(experiment_folder)

    return models_paths


############################## script ##############################
# Model Selection from directory
model_paths = get_model_path()

# Load the data