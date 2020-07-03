import glob
import random
import shutil
import os

# This script will split a folder with the following structure into (train,validation,test) sets:
# /data_folder
#   /lenses
#       lots of .fits files
#   /negatives
#       lots of .fits files
#   /sources
#       /1
#           1.fits
#       /2
#           2.fits
#       /3
#           3.fits
#       /4 etc up to 100.000 folders
#
# The same structure is wanted by then the /data_folder should either by train, validation, or test.
# 
###### FUNCTIONS ######


# Create target directory & all intermediate directories if don't exists
def create_dir_if_not_exists(dirName):
    try:
        os.makedirs(dirName)    
        print("Directory " , dirName ,  " Created ", flush=True)
    except FileExistsError:
        print("Directory " , dirName ,  " already exists", flush=True)


def splitInto_Train_Validation_Test(data_paths, train_frac, validation_frac):
    num_files = len(data_paths)
    train_paths      = data_paths[0 : int(train_frac*num_files)]
    validation_paths = data_paths[int(train_frac*num_files) : int(validation_frac*num_files + train_frac*num_files)]
    test_paths       = data_paths[int(validation_frac*num_files + train_frac*num_files): num_files]
    return train_paths, validation_paths, test_paths


# Move Sources data
def move_data(paths, set_type, data_class):
    for path in paths:
        parts = path.split("/")
        if data_class == 'sources':
            to_path = 'data1/' + set_type + '/' + data_class + '/' + parts[3] + "/"
        else:
            to_path = 'data1/' + set_type + '/' + data_class + '/'
        create_dir_if_not_exists(to_path)
        to_file_path = to_path + parts[-1]
        shutil.move(path, to_file_path)
######

# Parameters
train_frac      = 0.8
validation_frac = 0.1
test_frac       = 0.1
######

# The sum must be 1.0
assert (train_frac + validation_frac + test_frac) == 1.0

# Root dirs
path_sources   = "data/training/sources/"
path_lenses    = "data/training/lenses/"
path_negatives = "data/training/negatives/"

# Get all file names
data_paths_sources = glob.glob(path_sources + "*/*.fits")
data_paths_lenses = glob.glob(path_lenses + "*_r_*.fits")
data_paths_negatives = glob.glob(path_negatives + "*_r_*.fits")

print("num sources: {}".format(len(data_paths_sources)))
print("num lenses : {}".format(len(data_paths_lenses)))
print("num negativ: {}".format(len(data_paths_negatives)))

# Shuffle data
random.shuffle(data_paths_sources)
random.shuffle(data_paths_lenses)
random.shuffle(data_paths_negatives)

train_paths_S, validation_paths_S, test_paths_S = splitInto_Train_Validation_Test(data_paths_sources, train_frac, validation_frac)
train_paths_L, validation_paths_L, test_paths_L = splitInto_Train_Validation_Test(data_paths_lenses, train_frac, validation_frac)
train_paths_N, validation_paths_N, test_paths_N = splitInto_Train_Validation_Test(data_paths_negatives, train_frac, validation_frac)

# Lenses
move_data(paths=train_paths_L, set_type="train", data_class='lenses')
move_data(paths=validation_paths_L, set_type="validation", data_class='lenses')
move_data(paths=test_paths_L, set_type="test", data_class='lenses')

# Negatives
move_data(paths=train_paths_N, set_type="train", data_class='negatives')
move_data(paths=validation_paths_N, set_type="validation", data_class='negatives')
move_data(paths=test_paths_N, set_type="test", data_class='negatives')

# Sources
move_data(paths=train_paths_S, set_type="train", data_class='sources')
move_data(paths=validation_paths_S, set_type="validation", data_class='sources')
move_data(paths=test_paths_S, set_type="test", data_class='sources')

print("done")