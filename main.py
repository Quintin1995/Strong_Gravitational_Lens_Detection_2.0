from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from load_data import *
from utils import *
import random
import matplotlib.pyplot as plt
from Network import *

## Most, if not all of these 'hyper'-paramters should be read from a run.yaml file,
## which in turn creates a paramter class with all parameters as object properties.

##### Paths to data
lenses_path    = "data/training/lenses/"
negatives_path = "data/training/negatives/"
sources_path   = "data/training/sources/"

# Data type of images in the numpy array
data_type      = np.float32

# Image dimensionality
img_dims = (101,101,1)

# Whether to normalize (normalize per data array and not per image) the data during the image loading process.
normalize = "per_image"        #options = {"None", "per_image", "per_array"}

# Determines the splitting point of the data. Splitting percentage between test and train data.
test_fraction  = 0.2             # 0.2 means that 20% of the data will be reserved for test data.

# Alpha scaling, randomly drawn from this uniform distribution. Because the lensing features usually are of a lower luminosity than the LRG. Source scaling factor.
mock_lens_alpha_scaling = (0.02, 0.30)

# Whether you want to see plots and extra print output
verbatim = False

# Augmenation Parameters:
aug_zoom_range = (1.0,1.05)          # This range will be sampled from uniformly.
aug_num_pixels_shift_allowed = 4     # In Pixels
aug_rotation_range     = 360         # In Degrees
aug_do_horizontal_flip = True        # 50% of the time do a horizontal flip)  
aug_default_fill_mode  = 'nearest'   # Interpolation method, for data augmentation.
aug_width_shift_range  = aug_num_pixels_shift_allowed/img_dims[0]    # Fraction of width as allowed shift
aug_height_shift_range = aug_num_pixels_shift_allowed/img_dims[1]    # Fraction of height as allowed shift

# Network Paramters
net_name          = "resnet18"
net_learning_rate = 0.0001
net_model_metrics = "binary_accuracy"
net_num_outputs   = 1
net_epochs        = 1
net_batch_size    = 64

# Loading the input data
fraction_to_load_lenses    = 1.00    #range = [0,1]
fraction_to_load_negatives = 1.00    #range = [0,1]
fraction_to_load_sources   = 0.02     #range = [0,1]

# Chunk Paramters
num_chunks = 10    # Number of chunks to be generated 
chunksize  = 1280   # The number of images that will fit into one chunk

######### END PARAMS #########



###### Step 1.0: Create and store normalized data arrays.
lenses_array    = get_data_array(img_dims, path=lenses_path, fraction_to_load=fraction_to_load_lenses, are_sources=False, normalize=normalize)
negatives_array = get_data_array(img_dims, path=negatives_path, fraction_to_load=fraction_to_load_negatives, are_sources=False, normalize=normalize)
sources_array   = get_data_array(img_dims, path=sources_path, fraction_to_load=fraction_to_load_sources, are_sources=True, normalize=normalize)


###### Step 1.1: show some of the stored images from the data array to the user.
if verbatim:
    show_random_img_plt_and_stats(lenses_array,    num_imgs=1, title="lenses")
    show_random_img_plt_and_stats(negatives_array, num_imgs=1, title="negatives")
    show_random_img_plt_and_stats(sources_array,   num_imgs=1, title="sources")


###### Step 2: Split the data into train and test data.
X_train_lenses, X_test_lenses, y_train_lenses, y_test_lenses             = split_train_test_data(lenses_array, 0.0, test_fraction=test_fraction)
X_train_negatives, X_test_negatives, y_train_negatives, y_test_negatives = split_train_test_data(negatives_array, 0.0, test_fraction=test_fraction)
X_train_sources, X_test_sources, _, _                                    = split_train_test_data(sources_array, "no_label",  test_fraction=test_fraction)


###### Step 4.0 - Load a Training Chunk of data - This includes the merging of lenses with sources (creation of mock lenses).
X_train_chunk, y_train_chunk = load_chunk(chunksize, X_train_lenses, X_train_negatives, X_train_sources, data_type, mock_lens_alpha_scaling)
X_test_chunk, y_test_chunk   = load_chunk(chunksize, X_test_lenses, X_test_negatives, X_test_sources, data_type, mock_lens_alpha_scaling)


###### Step 4.1 - Sanity check of the train and test chunk
# 1: Are both positive and negative examples within the same brightness ranges?
# 2: I have added a per image normalization, because a couple of outliers ruin the normalization per data array (That is my hypothesis at least.)
if False:
    print(y_train_chunk)
    idxs_pos = np.where(y_train_chunk == 1.0)
    idxs_neg = np.where(y_train_chunk == 0.0)

    for i in range(25):
        pos_img = X_train_chunk[random.choice(list(idxs_pos[0]))]
        neg_img = X_train_chunk[random.choice(list(idxs_neg[0]))]
        show2Imgs(pos_img, neg_img, "pos max pixel: {0:.3f}".format(np.amax(pos_img)), "neg max pixel: {0:.3f}".format(np.amax(neg_img)))


###### Step 5.0 - Data Augmentation - Data Generator Keras
datagen = ImageDataGenerator(
        rotation_range=aug_rotation_range,
        width_shift_range=aug_width_shift_range,
        height_shift_range=aug_height_shift_range,
        zoom_range=aug_zoom_range,
        horizontal_flip=aug_do_horizontal_flip,
        fill_mode=aug_default_fill_mode,
        )

###### Step 6.0 - Create Neural Network - Resnet18
resnet18   = Network(net_name, net_learning_rate, net_model_metrics, img_dims, net_num_outputs)


###### Step 7.0 - Training
# datagen.fit(X_train_chunk)
for chunk_idx in range(num_chunks):
    X_train_chunk, y_train_chunk = load_chunk(chunksize, X_train_lenses, X_train_negatives, X_train_sources, data_type, mock_lens_alpha_scaling)
    start_time = time.time()
    resnet18.model.fit_generator(datagen.flow(X_train_chunk, y_train_chunk, batch_size=net_batch_size),
                        steps_per_epoch=len(X_train_chunk) / net_batch_size, epochs=net_epochs)
    print("Training on chunk took: {}".format(hms(time.time() - start_time)))
