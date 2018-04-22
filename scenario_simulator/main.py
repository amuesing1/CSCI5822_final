from __future__ import division
import h5py
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../tracking_scripts/modules/')
from detector import Detector
from modules.scenario import Scenario
#  from modules.detector import Detector
from modules.regions import Region
from modules.helper import visualizeRegion, checkScenarioAlreadyExists, checkDetectionExists

def create_scenario(h5_path):
    print("\tCreating Scenario...")

    # Define region names
    region_names = ["Boulder", "North Korea", "Israel"]

    # Define region image paths

    region_img_paths = ['imgs/boulder.png', 'imgs/north_korea.png', 'imgs/israel.png']

    # Define region coordinates
    region_coordinates = [{'latmin' : 39.972258, 'latmax' : 39.989349, 'lonmin' : -105.295428, 'lonmax' : -105.267471},
            			  {'latmin' : 39.490420, 'latmax' : 39.513332, 'lonmin' :  127.232188, 'lonmax' :  127.263597},
            			  {'latmin' : 32.077091, 'latmax' : 32.101197, 'lonmin' :   34.758563, 'lonmax' :   34.789610}]


    # Initialize REGIONS
    REGIONS = [Region(region_name, img_path, coordinates) for region_name, img_path, coordinates
                    in zip(region_names, region_img_paths, region_coordinates)]

    # Instantiate new scenario with regions
    new_scenario = Scenario(REGIONS, h5_path)

    # Generate data and create hdf5
    new_scenario.generateRegionData()

    # Write the h5 file
    new_scenario.writeH5()

    # Visualize a region, testing to see if it works correctly
    # new_scenario.visualizeRegion("north_korea")


def create_detector(h5_path):
    print("\tCreating Detector...")

    new_detector = Detector(h5_path)
    new_detector.readH5()
    new_detector.applyDetection()


def main(h5_path):

    # Create the scenario with multiple regions, each with a number of randomly generated dynamics models - save to H5
    # ------------------------------------------------------------------------------------------------------------------------
    # Check if scenario already exists
    write_scenario = checkScenarioAlreadyExists(h5_path)

    if write_scenario:
        create_scenario(h5_path)
    # ------------------------------------------------------------------------------------------------------------------------



    # Apply Detector - Add 3 Layers: Detections, Intensities, Centroids
    # ------------------------------------------------------------------------------------------------------------------------
    # Check if detection layer already exists
    apply_detector = checkDetectionExists(h5_path, "boulder")

    if apply_detector:
        create_detector(h5_path)
    # ------------------------------------------------------------------------------------------------------------------------



    # Check with visualization
    # ------------------------------------------------------------------------------------------------------------------------
    visualizeRegion(h5_path)
    # ------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    h5_path = "data/scenario_test.hdf5"

    main(h5_path)
