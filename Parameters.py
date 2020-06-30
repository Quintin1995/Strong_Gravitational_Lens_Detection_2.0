import os
from utils import *
import json

class Parameters(object):
    def __init__(self, settings):

        # Model Name
        self.model_name      = settings["model_name"]  # for example "first_model" must be something unique
        
        # Create Directories used throughout the project
        create_dir_if_not_exists("models")
        create_dir_if_not_exists("runs")
        create_dir_if_not_exists("slurms")


        ##### Paths to data
        self.lenses_path     = settings["lenses_path"]
        self.negatives_path  = settings["negatives_path"]
        self.sources_path    = settings["sources_path"]

        # Data type of images in the numpy array
        self.data_type = None

        # Image dimensionality
        self.img_dims = (settings["img_width"], settings["img_height"], settings["img_channels"])

        # Whether to normalize (normalize per data array and not per image) the data during the image loading process.
        self.normalize = settings["normalize"]        #options = {"None", "per_image", "per_array"}

        # Determines the splitting point of the data. Splitting percentage between test and train data.
        self.test_fraction = settings["test_fraction"]            # 0.2 means that 20% of the data will be reserved for test data.

        # Alpha scaling, randomly drawn from this uniform distribution. Because the lensing features usually are of a lower luminosity than the LRG. Source scaling factor.
        self.mock_lens_alpha_scaling = (settings["mock_lens_alpha_scaling_min"],settings["mock_lens_alpha_scaling_max"])

        # Whether you want to see plots and extra print output
        self.verbatim = settings["verbatim"]

        # Data Augmenation Paramters
        self.aug_zoom_range               = (settings["aug_zoom_range_min"], settings["aug_zoom_range_max"]) # This range will be sampled from uniformly.
        self.aug_num_pixels_shift_allowed = settings["aug_num_pixels_shift_allowed"]              # In Pixels
        self.aug_rotation_range           = settings["aug_rotation_range"]                        # In Degrees
        self.aug_do_horizontal_flip       = settings["aug_do_horizontal_flip"]                    # 50% of the time do a horizontal flip)  
        self.aug_default_fill_mode        = settings["aug_default_fill_mode"]                     # Interpolation method, for data augmentation.
        self.aug_width_shift_range        = self.aug_num_pixels_shift_allowed / self.img_dims[0]  # Fraction of width as allowed shift
        self.aug_height_shift_range       = self.aug_num_pixels_shift_allowed / self.img_dims[1]  # Fraction of height as allowed shift

        # Network Paramters
        self.net_name          = settings["net_name"]
        self.net_learning_rate = settings["net_learning_rate"]
        self.net_model_metrics = settings["net_model_metrics"]
        self.net_num_outputs   = settings["net_num_outputs"]
        self.net_epochs        = settings["net_epochs"]
        self.net_batch_size    = settings["net_batch_size"]

        # Loading the input data - What fraction of the data should be loaded into ram?
        self.fraction_to_load_lenses    = settings["fraction_to_load_lenses"]       # range = [0,1]
        self.fraction_to_load_negatives = settings["fraction_to_load_negatives"]    # range = [0,1]
        self.fraction_to_load_sources   = settings["fraction_to_load_sources"]      # range = [0,1]

        # Chunk Parameters
        self.num_chunks = settings["num_chunks"]  # Number of chunks to be generated 
        self.chunksize  = settings["chunksize"]   # The number of images that will fit into one chunk

        # Path to weights file for prediction (the referenced file should be a .h5 file that is trained)
        # self.full_path_predict_weights = os.path.join(settings["path_trained_folder"], settings["filename_trained_weights"])

        # Path stuff
        self.root_dir_models        = settings["root_dir_models"]
        self.model_folder           = get_time_string() + "_" +self.model_name   #A model will be stored in a folder with just a date&time as folder name
        self.model_path             = os.path.join(self.root_dir_models, self.model_folder)     #path of model
        self.make_model_dir()       #create directory for all data concerning this model.
        
        # Weights .h5 file
        self.weights_extension      = ".h5"                 #Extension for saving weights
        self.filename_weights       = self.model_name + "_weights_only" + self.weights_extension
        self.full_path_of_weights   = os.path.join(self.model_path, self.filename_weights)

        # Csv logger file to store the callback of the .fit function. It stores the history of the training session.
        self.history_extension      = ".csv"                 #Extension for history callback
        self.filename_history       = self.model_name + "_history" + self.history_extension
        self.full_path_of_history   = os.path.join(self.model_path, self.filename_history)

        #output path of .png
        self.figure_extension      = ".png"                 #Extension for figure 
        self.filename_figure       = self.model_name + "_results" + self.figure_extension
        self.full_path_of_figure   = os.path.join(self.model_path, self.filename_figure)

        # Output path of .json       dumps all parameters into a json file
        self.param_dump_extension  = ".json"                #Extension for the paramters being written to a file
        self.filename_param_dump   = self.model_name + "_param_dump" + self.param_dump_extension
        self.full_path_param_dump  = os.path.join(self.model_path, self.filename_param_dump)

        # Plot parameters
        self.chunk_plot_interval   = settings["chunk_plot_interval"]
        self.chunk_save_interval   = settings["chunk_save_interval"]

        # Validation_steps - Number of validation images that will be tested during training. (per chunk)
        self.validation_steps      = settings["validation_steps"]

        #store all parameters of this object into a json file
        self.write_parameters_to_file()


    # Create a folder where all model input/ouput is stored.
    def make_model_dir(self):
        try:
            os.mkdir(self.model_path)
        except OSError:
            print ("Creation of the directory %s failed" % self.model_path, flush=True)
        else:
            print ("Successfully created the directory: %s " % self.model_path, flush=True)


    # Write all the paramters defined in parameters class to a file
    def write_parameters_to_file(self):
        with open(self.full_path_param_dump, 'w') as outfile:
            json_content = self.toJSON()
            outfile.write(json_content)
            print("Wrote all run parameters to directory: {}".format(self.full_path_param_dump), flush=True)


    # Dump all object properties to a .json file.
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True,indent=4)
