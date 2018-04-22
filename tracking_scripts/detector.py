'''
***************************************************
File: detection.py

Getting detections from noisey fake data

***************************************************
'''

from __future__ import division
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import matplotlib.image as mgimg
import imageio
import warnings
import matplotlib.cbook
from scipy import ndimage
sys.path.append('../../scenario_simulator/modules/')
from modules.helper import loadingBar

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

__author__ = "Jeremy Muesing"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Luke Burks", "Jeremy Muesing", "Michael Iuzzolino"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jeremy Muesing"
__email__ = "jeremy.muesing@colorado.edu"
__status__ = "Development"


VIS_DATA = True
CREATE_GIF = False


class Detector(object):
    def __init__(self, h5_path, detection_bypass = True):
        self.detection_threshold = 30
        self.h5_path = h5_path
        self.detection_bypass=detection_bypass
        self.detector_layers = ["Detections", "Centroids", "Intensities"]

        # Initialize h5 data
        self.h5_data = {}

    def readH5(self):
        print("\tReading h5 file...")
        # Read data
        with h5py.File(self.h5_path, 'r') as infile:
            for region_name, region_info in infile.items():

                # Get region coordinates
                # ------------------------------------------------------------
                region_coordinates = {}
                for attr_name, attr_val in infile[region_name].attrs.items():
                    region_coordinates[attr_name] = attr_val
                # ------------------------------------------------------------

                # Initialize h5 data with region name
                self.h5_data[region_name] = None

                # Get region group and declare subgrounds for core layers
                # ------------------------------------------------------------
                group = infile[region_name]
                gt_subgroup = group["GroundTruth"]
                sb_subgroup = group["ShakeBase"]
                pb_subgroup = group["PixelBleed"]
                # ------------------------------------------------------------


                # Core subgroups
                # ------------------------------------------------------------
                groundTruthData = np.array([np.array(val) for val in gt_subgroup.values()])
                shakeBaseData = np.array([np.array(val) for val in sb_subgroup.values()])
                pixelBleedData = np.array([np.array(val) for val in pb_subgroup.values()])
                # ------------------------------------------------------------


                # Noise Layers
                # ------------------------------------------------------------
                noise = np.array(group["Noise"])
                structNoise = np.array(group['StructuredNoise'])
                shotNoise = np.array(group['ShotgunNoise'])
                # ------------------------------------------------------------


                # Consolidate gt, sb, and pb
                # ------------------------------------------------------------
                groundTruth = np.sum(groundTruthData, axis=0)
                shakeBase = np.sum(shakeBaseData, axis=0)
                pixelBleed = np.sum(pixelBleedData, axis=0)
                # ------------------------------------------------------------


                # HDF5 parameters
                # ------------------------------------------------------------
                nFrames = noise.shape[0]
                nRows = noise.shape[1]
                nCols = noise.shape[2]
                # ------------------------------------------------------------


                # Generate Full Set (aggregate)
                # ------------------------------------------------------------
                full_set = np.zeros(shape=(nFrames, nRows, nCols))
                for i in range(nFrames):
                    if self.detection_bypass:
                        full_set[i]=groundTruth[i]
                    else:
                        full_set[i] = groundTruth[i]+shakeBase[i]+noise[i]+pixelBleed[i]+shotNoise[i]
                # ------------------------------------------------------------

                # Add region to data dict
                # ------------------------------------------------------------
                self.h5_data[region_name] = {"raw_data" : full_set, "coordinates" : region_coordinates}
                # ------------------------------------------------------------

    def applyDetection(self):
        for region_name, region_items in self.h5_data.items():
            print("\n\tRegion: {}".format(region_name))
            print("\t{}".format("-"*len("Region: {}".format(region_name))))

            region_data = region_items["raw_data"]
            region_coords = region_items["coordinates"]
            self.regionDetector(region_name, region_data)
            self.centroidDetector(region_name, region_coords)
            self.writeRegionToH5(region_name)
            print("\n")


    def regionDetector(self, region_name, region_data):
        print("\tGenerating Detection Layer...")
        if self.detection_bypass:
            self.h5_data[region_name]["detection_data"] = region_data
        else:
            # H5 parameters
            nFrames = region_data.shape[0]
            nRows = region_data.shape[1]
            nCols = region_data.shape[2]

            # Init data average and difference filter objects
            data_avg = np.zeros(shape=(nRows, nCols))
            difference_filter = np.zeros(shape=(nRows, nCols))
            data_surround_avg = np.zeros(shape=(nRows, nCols))
            detection_data = np.copy(region_data)

            for i in range(nFrames):
                loadingBar(i, nFrames)

                # Calculate data median at frame i
                data_i_median = np.median(region_data[i])

                for j in range(nRows):
                    for k in range(nCols):
                        if ((j-3 < 0) or (j+3 > nRows) or (k-3 < 0) or (k+3 > nCols)):
                            data_surround_avg[j][k] =  data_i_median
                            data_avg[j][k] = data_i_median
                        else:
                            data_surround_array = np.copy(region_data[i, j-3:j+4, k-3:k+4])
                            data_surround_array[1:6, 1:6] = 0
                            data_surround_avg[j][k]= np.sum(data_surround_array) / 24
                            data_avg[j][k] = np.mean(region_data[i, j-2:j+3, k-2:k+3])

                        difference_filter[j][k] = np.abs(data_avg[j][k] - data_surround_avg[j][k])

                mdev = np.median(difference_filter)
                outlier_frame = difference_filter / (mdev + 1e-7)

                detection_data[i][np.where(outlier_frame < self.detection_threshold)] = 0
                for j in range(nRows):
                    for k in range(nCols):
                        if outlier_frame[j][k] > self.detection_threshold:
                            detection_data[i, j-2:j+3, k-2:k+3] = np.copy(region_data[i, j-2:j+3, k-2:k+3])

            # Add detection data to h5 data region layer
            self.h5_data[region_name]["detection_data"] = detection_data


    def centroidDetector(self, region_name, region_coords):
        """
            Finds the intensity weighted centroid of all detections
        """

        print("\tGenerating Centroid and Intensities Layers...")

        # Initialize detection data from region
        detection_data = self.h5_data[region_name]["detection_data"]

        # Initialize lat/lon minimums and maximums
        latmin = region_coords['latmin']
        latmax = region_coords['latmax']
        lonmin = region_coords['lonmin']
        lonmax = region_coords['lonmax']


        # Initialize containers for centroids and intensities
        region_centroids = []
        region_centroid_intensities = []

        # Iterate through detection data frames and find centroids in each frame
        nFrames = detection_data.shape[0]
        for i in range(nFrames):

            # Show processing progress
            loadingBar(i, nFrames)


            # Flip detection data
            detection_copy_i = np.flip(detection_data[i], 0)


            # Classify the objects detected in the image
            label_matrix, num_centroids = ndimage.measurements.label(detection_copy_i)



            # Init containers for centroids and intensities
            centroids_on_frame = []
            intensities_on_frame = []
            for cluster_index in range(1, num_centroids+1):

                # Get Centroid's POSITION
                # --------------------------------------------------------------------------------------------------
                x = np.where(label_matrix == cluster_index)[0]  #  latitude pixel
                y = np.where(label_matrix == cluster_index)[1]  #  longitude pixel

                values = np.copy(detection_copy_i[x, y])


                cluster_sum = ndimage.sum(detection_copy_i, label_matrix, np.arange(1, num_centroids+1))[cluster_index-1]

                # Calculate averages
                latitude_pixelspace = sum(x * values) / cluster_sum    # latitude pixelspace
                longitude_pixelspace = sum(y * values) / cluster_sum    # longitude pixelspace

                # Lat/lon deg to pix
                lat_deg_to_pixel = np.abs(latmax - latmin) / detection_data.shape[1]
                lon_deg_to_pixel = np.abs(lonmax - lonmin) / detection_data.shape[0]

                # Map from pixelspace to lat/lon coordinates
                lat_bar = latmax - (latitude_pixelspace * lat_deg_to_pixel)
                lon_bar = lonmin + (longitude_pixelspace * lon_deg_to_pixel)


                # ******************************************************************************************
                #           DEBUGGING
                # ******************************************************************************************
                # print("\n\n")
                # print("-"*50)
                #
                # print("LONGITUDE")
                # print("---------")
                # print("lon min: {}".format(lonmin))
                # print("lon max: {}".format(lonmax))
                #
                # print("\n")
                #
                # print("latitude_pixelspace: {}".format(latitude_pixelspace))
                # print("lon_bar: {}".format(lon_bar))
                # print("\n\n")
                #
                # print("LATITUDE")
                # print("--------")
                # print("lat min: {}".format(latmin))
                # print("lat max: {}".format(latmax))
                #
                # print("\n")
                #
                # print("longitude_pixelspace: {}".format(longitude_pixelspace))
                # print("lat_bar: {}".format(lat_bar))
                # print("\n\n")
                #
                #
                #
                # plt.close()
                # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                # ax1.imshow(detection_data[i])
                # ax2.imshow(detection_copy_i)
                #
                # ax3.imshow(label_matrix)
                #
                #
                # plt.show()
                # raw_input("pause.")
                # ******************************************************************************************





                # Truncate to 6 decimals
                lat_bar = float('%.6f' % round(lat_bar, 6))
                lon_bar = float('%.6f' % round(lon_bar, 6))



                # Update frame centroids
                centroids_on_frame.append([lat_bar, lon_bar])
                # --------------------------------------------------------------------------------------------------

                # Get Centroid's INTENSITY
                # --------------------------------------------------------------------------------------------------
                if self.detection_bypass:
                    #  print values
                    intensity=max([0],values[0]+np.random.normal(0,2))
                else:
                    intensity = ndimage.mean(detection_data[i], label_matrix, np.arange(1, num_centroids+1))[cluster_index-1]
                intensities_on_frame.append([intensity]) # Add as a list for consistency in downstream code - MUST KEEP THIS WAY or it breaks!
                # --------------------------------------------------------------------------------------------------

            # Update region centroids
            region_centroids.append(centroids_on_frame)
            region_centroid_intensities.append(intensities_on_frame)

        self.h5_data[region_name]["Centroids"] = region_centroids
        self.h5_data[region_name]["Intensities"] = region_centroid_intensities




    def writeRegionToH5(self, region_name):
        print("\tWriting detections to h5...")
        # Write Data
        with h5py.File(self.h5_path, 'a') as outfile:

            # Set region group
            region_group = outfile[region_name]

            # Clean up previos detector layers if present
            # Check if filter already exists; if so, remove it.
            # ------------------------------------------------------------------------------------------------------------
            for detection_layer in self.detector_layers:
                if detection_layer in region_group.keys():
                    del region_group[detection_layer]
                    print("\t* Overwriting previous {} filter...".format(detection_layer))
            # ------------------------------------------------------------------------------------------------------------

            # Create DETECTIONS layer
            # ------------------------------------------------------------------------------------------------------------
            detection_data = self.h5_data[region_name]["detection_data"]
            region_group.create_dataset('Detections', data=detection_data)
            # ------------------------------------------------------------------------------------------------------------

            # Create CENTROIDS layer
            # ------------------------------------------------------------------------------------------------------------
            # Issue with just using create_dataset: TypeError: Object dtype dtype('O') has no native HDF5 equivalent
            # Solution Reference: https://stackoverflow.com/questions/37214482/saving-with-h5py-arrays-of-different-sizes
            centroid_group = region_group.create_group("Centroids")
            centroids_data = self.h5_data[region_name]["Centroids"]

            # Add attrs to centroid_group
            centroid_group.attrs["format"] = "[lat, lon]"

            # Encode the data
            for frame_i in range(len(centroids_data)):
                frame_centroids = centroids_data[frame_i]

                for centroid_i in range(len(frame_centroids)):
                    frame_centroid = frame_centroids[centroid_i]

                    dataset_name = "frame_{}_centroid_{}".format(frame_i, centroid_i)
                    centroid_group.create_dataset(dataset_name, data=frame_centroid)
            # ------------------------------------------------------------------------------------------------------------


            # Create INTENSITIES layer
            # ------------------------------------------------------------------------------------------------------------
            intensity_group = region_group.create_group("Intensities")
            intensities_data = self.h5_data[region_name]["Intensities"]

            # Encode the data
            for frame_i in range(len(intensities_data)):
                frame_intensities = intensities_data[frame_i]

                for intensity_i in range(len(frame_intensities)):
                    frame_intensity = frame_intensities[intensity_i]

                    dataset_name = "frame_{}_intensity_{}".format(frame_i, intensity_i)
                    intensity_group.create_dataset(dataset_name, data=frame_intensity)
            # ------------------------------------------------------------------------------------------------------------


def module_tester(h5_path):
    """
        FOR TESTING ONLY
    """
    new_detector = Detector(h5_path)
    new_detector.readH5()
    new_detector.applyDetection()


if __name__ == '__main__':
    h5_path = "../data/scenario_test.hdf5"  # add ../ for module testing
    module_tester(h5_path)
