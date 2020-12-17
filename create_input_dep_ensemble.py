import argparse


############################################################ Functions ############################################################


# Deal with input arguments
def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_name", help="Name of the ensemble. There will be a folder with the given name.", default="test_ensemble", required=False)
    # parser.add_argument("--sample_size", help="The amount of images used in the validation set to optimize ensemble model weights. A maximum of 551 can be used.", default=1024, required=False)
    # parser.add_argument("--method", help="What ensemble method should be used? To determine model weights? For example: Nelder-Mead.", default="Nelder-Mead", required=False)
    parser.add_argument("--network", help="Name of the network used, in order to load the right architecture", default="simple_net", required=False)
    args = parser.parse_args()
    return args



def main():
    args = _get_arguments()
    print(args)

    # Load train data
    # Load validation data
    # Load test data

    # Determine which models are going to be members for the ensemble.

    # Construct input dependent ensemble model

 

############################################################ Script    ############################################################




if __name__ == "__main__":
    main()