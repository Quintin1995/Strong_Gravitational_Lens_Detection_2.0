import argparse
from DataGenerator import DataGenerator
from Network import Network
import numpy as np
from Parameters import Parameters
from utils import load_settings_yaml, show_random_img_plt_and_stats
import matplotlib.pyplot as plt


# Proof of concept of surface brightness contouring



# Define ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("--run", help="Location/path of the run.yaml file. This is usually structured as a path.", default="runs/run_poc_sbc.yaml", required=False)
args = parser.parse_args()

# Unpack args
yaml_path = args.run

# Load all settings from .yaml file
settings_yaml = load_settings_yaml(yaml_path)                                           # Returns a dictionary object.
params = Parameters(settings_yaml, yaml_path)
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.

# Create Custom Data Generator
dg = DataGenerator(params)

X_train_chunk, y_train_chunk = dg.load_chunk(params.chunksize, dg.Xlenses_train, dg.Xnegatives_train, dg.Xsources_train, params.data_type, params.mock_lens_alpha_scaling)

#only take lenses
count = 25
x = X_train_chunk[0:count]
y = y_train_chunk[0:count]


plt.imshow(np.squeeze(x[0]), origin='lower', interpolation='none', cmap=plt.cm.binary)

gaus_kernel = np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]])

from scipy import ndimage

for i in range(x.shape[0]):
    img = np.squeeze(x[i])
    x[i] = np.expand_dims(ndimage.convolve(img, gaus_kernel, mode='constant', cval=0.0), axis = 2)


plt.imshow(np.squeeze(x[0]), origin='lower', interpolation='none', cmap=plt.cm.binary)
plt.show()

x=2

