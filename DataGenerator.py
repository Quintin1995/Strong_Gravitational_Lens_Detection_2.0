class DataGenerator:
    def __init__(self, params, *args, **kwargs):
        self.params = params

        self.PSF_r = self.compute_PSF_r()

    
    
    def compute_PSF_r(self):
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

        seds = np.loadtxt("data/SED_colours_2017-10-03.dat")

        Rg = 3.30
        Rr = 2.31
        Ri = 1.71
        return PSF_r
        ### END OF IMPORTANT PIECE.