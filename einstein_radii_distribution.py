import os
import glob
from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyfits
from scipy import ndimage
import scipy

def compute_PSF_r():
    ## This piece of code is needed for some reason that i will try to find out later.
    nx = 101
    ny = 101
    f1 = pyfits.open("data/PSF_KIDS_175.0_-0.5_r.fits")  # PSF
    d1 = f1[0].data
    d1 = np.asarray(d1)
    nx_, ny_ = np.shape(d1)
    PSF_r = np.zeros((nx, ny))  # output
    dx = (nx - nx_) // 2  # shift in x
    dy = (ny - ny_) // 2  # shift in y
    for ii in range(nx_):  # iterating over input array
        for jj in range(ny_):
            PSF_r[ii + dx][jj + dy] = d1[ii][jj]

    return PSF_r

def get_empty_dataframe():
    # Define which parameter to collect
    column_names = ["LENSER", "LENSAR", "LENSAA", "LENSSH", "LENSSA", "SRCER", "SRCX", "SRCY", "SRCAR", "SRCAA", "SRCSI", "path"]

    # Create a dataframe for parameter storage
    return pd.DataFrame(columns = column_names)


# Fill the dataframe with SIE parameters and Sersic parameters
# use numpy arrays for speed
def fill_dataframe(df, paths):

    # Define numpy arrays to temporarily hold the parameters
    # LENS PARAMETERS
    LENSER = np.zeros((len(paths),), dtype=np.float32)
    LENSAR = np.zeros((len(paths),), dtype=np.float32)
    LENSAA = np.zeros((len(paths),), dtype=np.float32)
    LENSSH = np.zeros((len(paths),), dtype=np.float32)
    LENSSA = np.zeros((len(paths),), dtype=np.float32)
    # SERSIC PARAMETERS
    SRCER = np.zeros((len(paths),), dtype=np.float32)
    SRCX = np.zeros((len(paths),), dtype=np.float32)
    SRCY = np.zeros((len(paths),), dtype=np.float32)
    SRCAR = np.zeros((len(paths),), dtype=np.float32)
    SRCAA = np.zeros((len(paths),), dtype=np.float32)
    SRCSI = np.zeros((len(paths),), dtype=np.float32)

    # Loop over all sources files
    for idx, filename in enumerate(paths):
        if idx % 1000 == 0:
            print("processing source idx: {}".format(idx))
        hdul = fits.open(filename)

        LENSER[idx] = hdul[0].header["LENSER"] 
        LENSAR[idx] = hdul[0].header["LENSAR"] 
        LENSAA[idx] = hdul[0].header["LENSAA"] 
        LENSSH[idx] = hdul[0].header["LENSSH"] 
        LENSSA[idx] = hdul[0].header["LENSSA"] 

        SRCER[idx] = hdul[0].header["SRCER"] 
        SRCX[idx] = hdul[0].header["SRCX"] 
        SRCY[idx] = hdul[0].header["SRCY"] 
        SRCAR[idx] = hdul[0].header["SRCAR"] 
        SRCAA[idx] = hdul[0].header["SRCAA"] 
        SRCSI[idx] = hdul[0].header["SRCSI"]

    df["LENSER"] = LENSER
    df["LENSAR"] = LENSAR
    df["LENSAA"] = LENSAA
    df["LENSSH"] = LENSSH
    df["LENSSA"] = LENSSA

    df["SRCER"] = SRCER
    df["SRCX"] = SRCX
    df["SRCY"] = SRCY
    df["SRCAR"] = SRCAR
    df["SRCAA"] = SRCAA
    df["SRCSI"] = SRCSI

    df["path"] = paths
    return df

##################### script #####################

# Set paths
path = os.path.join("data", "train", "sources")
path = os.path.join("data", "small_sources_set")

# Find all relevant files in subfolders
paths = glob.glob(os.path.join(path, "*/*.fits"))

# Initilize and fill dataframe
df = get_empty_dataframe()
df = fill_dataframe(df, paths)
print(df.head(5))

# What is the maximum Einstein Radius?
LENSER = list(df['LENSER'])
print("Maximum Einstein Radius: {}".format(max(LENSER)))

# And what does this source look like?
max_row = df.iloc[df['LENSER'].argmax()]
print(max_row)
print(type(max_row))
img = fits.getdata(max_row["path"]).astype(np.float32)
PSF_r = compute_PSF_r()
img = scipy.signal.fftconvolve(img, PSF_r, mode="same")
plt.imshow(img, origin='lower', interpolation='none', cmap=plt.cm.binary)
plt.show()

# What is the size of this sources in terms of pixels?

# plot distribution of certain parameter in dataframe


hfjdks=5