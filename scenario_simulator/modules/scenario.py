import numpy as np
import matplotlib.pyplot as plt
import h5py


class Scenario(object):

    def __init__(self, regions, h5_path):
        self.regions = regions
        self.h5_path = h5_path


    def writeH5(self):
        with h5py.File(self.h5_path, 'w') as outFile:

            # Set region img properties (nframes, ncols, nrows)
            outFile.attrs["region_frame_properties"] = self.regions[0].img_shape

            for region in self.regions:
                region.writeRegionH5(outFile)


    def generateRegionData(self):
        for region in self.regions:
            region.initPointTargets()
            region.generateLayers()


    




def module_tester():
    pass


if __name__ == '__main__':
    module_tester()
