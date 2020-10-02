import os
from utils import create_dir_if_not_exists, get_time_string
import json
from shutil import copyfile

class Parameters(object):
    def __init__(self, settings, yaml_fpath, mode="training"):

        # Model Name
        self.model_name      = settings["model_name"]  # for example "first_model" must be something unique
        
        # Create Directories used throughout the project
        create_dir_if_not_exists("models")
        create_dir_if_not_exists("runs")
        create_dir_if_not_exists("slurms")
        

        ##### Paths to data
        self.lenses_path_train     = settings["lenses_path_train"]
        self.negatives_path_train  = settings["negatives_path_train"]
        self.sources_path_train    = settings["sources_path_train"]

        self.lenses_path_validation     = settings["lenses_path_validation"]
        self.negatives_path_validation  = settings["negatives_path_validation"]
        self.sources_path_validation    = settings["sources_path_validation"]

        # Data type of images in the numpy array
        self.data_type = None

        # Image dimensionality
        self.img_dims = (settings["img_width"], settings["img_height"], settings["img_channels"])

        # Whether to normalize (normalize per data array and not per image) the data during the image loading process.
        self.normalize = settings["normalize"]        #options = {"None", "per_image", "per_array"}

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
        self.net_loss_function = settings["net_loss_function"]
        self.net_num_outputs   = settings["net_num_outputs"]
        self.net_epochs        = settings["net_epochs"]
        self.net_batch_size    = settings["net_batch_size"]
        self.es_patience       = settings["es_patience"]

        # Loading the input data - What fraction of the data should be loaded into ram?
        self.fraction_to_load_lenses_train    = settings["fraction_to_load_lenses_train"]     # range = [0,1]
        self.fraction_to_load_negatives_train = settings["fraction_to_load_negatives_train"]  # range = [0,1]
        self.fraction_to_load_sources_train   = settings["fraction_to_load_sources_train"]    # range = [0,1]

        self.fraction_to_load_lenses_vali    = settings["fraction_to_load_lenses_vali"]       # range = [0,1]
        self.fraction_to_load_negatives_vali = settings["fraction_to_load_negatives_vali"]    # range = [0,1]
        self.fraction_to_load_sources_vali   = settings["fraction_to_load_sources_vali"]      # range = [0,1]

        # Chunk Parameters
        self.num_chunks = settings["num_chunks"]  # Number of chunks to be generated 
        self.chunksize  = settings["chunksize"]   # The number of images that will fit into one chunk

        # Path stuff
        self.root_dir_models        = settings["root_dir_models"]
        self.model_folder           = get_time_string() + "_" +self.model_name   #A model will be stored in a folder with just a date&time as folder name
        self.model_path             = os.path.join(self.root_dir_models, self.model_folder)     #path of model
        if mode == "training":
            self.make_model_dir()       #create directory for all data concerning this model.
            create_dir_if_not_exists(os.path.join(self.model_path, "checkpoints"))
        
        # Weights .h5 file
        self.weights_extension      = ".h5"                 #Extension for saving weights
        self.filename_weights       = self.model_name + "_weights_only" + self.weights_extension
        self.full_path_of_weights   = os.path.join(self.model_path, self.filename_weights)

        self.set_checkpoint_properties()

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

        # In this folder a complete set of the model is stored, so that i can be rebuild later. This includes where it is in training stage (learning rate etc.)
        self.saved_model_folder      = "my_model_storage"
        self.full_path_model_storage = os.path.join(self.model_path, self.saved_model_folder)

        # In this txt a complete printout of the neural net architecture is stored.
        self.neural_net_txt_printout = "neural_net_printout.txt"
        self.full_path_neural_net_printout = os.path.join(self.model_path, self.neural_net_txt_printout)

        # Plot parameter
        self.chunk_plot_interval   = settings["chunk_plot_interval"]
        
        # Save interval
        self.chunk_save_interval   = settings["chunk_save_interval"]

        # Validation chunk size - Number of validation images that will be tested during training. (per chunk)
        self.validation_chunksize  = settings["validation_chunksize"]

        # EXPERIMENT PARAMTERS
        self.use_avg_pooling_2D    = settings["use_avg_pooling_2D"]

        self._set_segmentation_parameters(settings)
        
        if mode == "training":
            #copy run.yaml to model folder
            copyfile(yaml_fpath, os.path.join(self.model_path, "run.yaml"))

            #store all parameters of this object into a json file
            self._write_parameters_to_file()


    # This function ensures that older run.yaml(s) are compatible with the newer ones.
    def _set_segmentation_parameters(self, settings):
        if "do_max_tree_seg" not in settings:
            self.do_max_tree_seg       = False
        else:
            self.do_max_tree_seg       = settings["do_max_tree_seg"]            #Boolean, Whether to perform max_tree_segmenation or not.
        
        ## define kernel type and size
        if "conv_method" not in settings:
            self.conv_method           = "gaussian"
        else:
            self.conv_method           = settings["conv_method"]                #string# options= {"gaussian", "boxcar"}
        if "ksize" not in settings:
            self.ksize                 = 5
        else:
            self.ksize                 = settings["ksize"]                      #int#    options = {3,5}     If 3, it creates a 3x3 kernel. If 5, then it will create a 5x5 kernel
        
        ## Define cropping type and size
        if "do_square_crop" not in settings:
            self.do_square_crop        = False
        else:
            self.do_square_crop        = settings["do_square_crop"]             #Boolean, Whether to take a centered square crop
        if "do_circular_crop" not in settings:
            self.do_circular_crop      = False
        else:
            self.do_circular_crop      = settings["do_circular_crop"]           #Boolean, Whether to perform a circular cropping around the centre of the image
        if "crop_size" not in settings:
            self.crop_size             = 74
        else:
            self.crop_size             = settings["crop_size"]                  #int, size of the square/circular crop if enabled. (74,74) for example
        
        ## Define scaling factor
        if "do_scale" not in settings:
            self.do_scale              = False
        else:
            self.do_scale              = settings["do_scale"]                   #Boolean, Whether to perform brighness scaling of a pixel based on the distance from the centre of the image. The furter away the object the closer to zero its value will become.
        if "x_scale" not in settings:
            self.x_scale               = 30
        else:
            self.x_scale               = settings["x_scale"]                    #int # Scale brightness of pixel based on the distance from centre of the image   # According to the following formula e^(-distance/x_scale), which is exponential decay
        
        ## Define floodfill tolerance
        if "do_floodfill" not in settings:
            self.do_floodfill          = False
        else:
            self.do_floodfill          = settings["do_floodfill"]               #Boolean, to perform the floodfill operation after brightness distance scaling
        if "tolerance" not in settings:
            self.tolerance             = 20
        else:
            self.tolerance             = settings["tolerance"]                  #int, The allowed range of pixel intensity/brighness difference. The higher the value the more pixels will be floodfilled, and the smaller the segments will become.
        
        ## Filter max-tree based on node area size.
        if "area_th" not in settings:
            self.area_th               = 45
        else:
            self.area_th               = settings["area_th"]                    #int, Allowed minimum size of a node in the max-tree datastructure. If a certain structure in a max-tree-node has a smaller size, it is discarted.
        if "use_seg_imgs" not in settings:
            self.use_seg_imgs          = False
        else:
            self.use_seg_imgs          = settings["use_seg_imgs"]               #boolean# (if False, then the segmentated image will be used as mask and the original image will be used.) (If True, then the segmented images are used instead of the original images. (so, this means no masking))


    # Model Storage with checkpoints and a reference with a .yaml file, from which epoch the model is.
    def set_checkpoint_properties(self):
        # Best validation loss Weights .h5 file
        self.filename_weights_loss         = self.model_name + "_best_val_loss" + self.weights_extension
        self.full_path_of_weights_loss     = os.path.join(self.model_path, "checkpoints", self.filename_weights_loss)

        # Best validation metric Weights .h5 file (metric can be binary accuracy or f1 score or the fbeta score.)
        self.filename_weights_metric       = self.model_name + "_best_val_metric" + self.weights_extension
        self.full_path_of_weights_metric   = os.path.join(self.model_path, "checkpoints", self.filename_weights_metric)
        
        # Best validation loss .yaml file, stores loss and epoch number
        self.filename_yaml_loss            = self.model_name + "_best_val_loss.yaml"
        self.full_path_of_yaml_loss        = os.path.join(self.model_path, "checkpoints", self.filename_yaml_loss)

        # Best validation metric .yaml, stores score and epoch number
        self.filename_yaml_metric          = self.model_name + "_best_val_metric.yaml"
        self.full_path_of_yaml_metric      = os.path.join(self.model_path, "checkpoints", self.filename_yaml_metric)


    # Create a folder where all model input/ouput is stored.
    def make_model_dir(self):
        try:
            os.mkdir(self.model_path)
        except OSError:
            print ("Creation of the directory %s failed" % self.model_path, flush=True)
        else:
            print ("Successfully created the directory: %s " % self.model_path, flush=True)


    # Write all the paramters defined in parameters class to a file
    def _write_parameters_to_file(self):
        with open(self.full_path_param_dump, 'w') as outfile:
            json_content = self.toJSON()
            outfile.write(json_content)
            print("Wrote all run parameters to directory: {}".format(self.full_path_param_dump), flush=True)


    # Dump all object properties to a .json file.
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True,indent=4)
