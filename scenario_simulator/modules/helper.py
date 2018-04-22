import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

def checkScenarioAlreadyExists(h5_path):
    try:
        with h5py.File(h5_path, "r") as infile:
            h5_already_exists = True
    except IOError:
        h5_already_exists = False
        rewrite_file = True

    if h5_already_exists:
        valid_input = False
        while not valid_input:
            print("Scenario file already exists. Would you like to rewrite (R) the file or continue (C) to detections?")
            rewrite_file = str(raw_input(" >> "))

            if rewrite_file.lower() == 'r':
                double_check_delete = doubleCheckInput("Are you sure you want to erase the scenario")

                if double_check_delete:
                    print("\tPrepairing to rewrite file...\n")
                    rewrite_file = True
                    valid_input = True

            elif rewrite_file.lower() == 'c':
                print("\tMoving to detections...\n")
                rewrite_file = False
                valid_input = True
            else:
                print("Invalid input. Try again.")

    return rewrite_file


def checkDetectionExists(h5_path, test_region_name):

    detector_layers = ["Detections", "Centroids", "Intensities"]

    all_dector_layers_present = True
    with h5py.File(h5_path, "r") as infile:
        for region_name in infile.keys():
            for detection_layer in detector_layers:
                try:
                    infile[region_name][detection_layer]
                except KeyError:
                    all_dector_layers_present = False
                    reapply_detections = True
                    break

            if not all_dector_layers_present:
                break

    if all_dector_layers_present:
        valid_input = False
        while not valid_input:
            print("Detection layers already exist. Would you like to reapply (R) the detector or continue (C) to visualizations?")
            reapply_detections = str(raw_input(" >> "))

            if reapply_detections.lower() == 'r':

                # DOUBLE CHECK that the user wants to delete layers
                double_check_delete = doubleCheckInput("Are you sure you want to remove the detector's layers")

                if double_check_delete:
                    print("\tPrepairing to reapply detector...\n")
                    removeDetectorLayers(h5_path, detector_layers)
                else:
                    continue

                reapply_detections = True
                valid_input = True

            elif reapply_detections.lower() == 'c':
                print("\tMoving to visualization...\n")
                reapply_detections = False
                valid_input = True
            else:
                print("Invalid input. Try again.")

    return reapply_detections



def doubleCheckInput(message):
    valid_input = False
    while not valid_input:
        print("{} (y/n)?".format(message))
        double_check_delete = str(raw_input(" >> "))
        if double_check_delete.lower() == 'y':
            double_check_delete = True
            valid_input = True
        elif double_check_delete.lower() == 'n':
            double_check_delete = False
            valid_input = True
        else:
            continue

    return double_check_delete


def removeDetectorLayers(h5_path, detector_layers):

    with h5py.File(h5_path, "r+") as infile:
        for region_name in infile.keys():
            for detector_layer in detector_layers:
                try:
                    print("\t\t ** Removing {}'s {} layer".format(region_name, detector_layer))
                    del infile[region_name][detector_layer]
                except KeyError:
                    pass
            print("\n")



def visualizeRegion(h5_file):

    show_vis = doubleCheckInput("Show the visualization")

    if show_vis:
        visualizing_regions = True

        # Get Regions
        with h5py.File(h5_file, 'r') as infile:
            regions = infile.keys()
    else:
        print("Farewell.")
        return


    while visualizing_regions:

        valid_input = False
        while not valid_input:
            print("Choose a region: {}".format(regions))
            region_name = str(raw_input(" >> "))

            if region_name.lower() == 'q':
                visualizing_regions = False
                break

            elif region_name in regions:
                print("Visualizing {}...".format(region_name))
                valid_input = True

            else:
                print(" ** Invalid choice. Try again.")

        # Read data
        with h5py.File(h5_file, 'r') as infile:
            group = infile[region_name]
            gt_subgroup = group["GroundTruth"]
            sb_subgroup = group["ShakeBase"]
            pb_subgroup = group["PixelBleed"]

            # Core subgroups
            groundTruthData = np.array([np.array(val) for val in gt_subgroup.values()])
            shakeBaseData = np.array([np.array(val) for val in sb_subgroup.values()])
            pixelBleedData = np.array([np.array(val) for val in pb_subgroup.values()])

            # Noise Layers
            noise = np.array(group["Noise"])
            structNoise = np.array(group['StructuredNoise'])
            shotNoise = np.array(group['ShotgunNoise'])

            # Detections
            detections = np.array(group["Detections"])

            # Intensities
            intensities = np.array(group["Intensities"])

            # Centroids
            centroids = np.array(group["Centroids"])


        # Consolidate gt, sb, and pb
        groundTruth = np.sum(groundTruthData, axis=0)
        shakeBase = np.sum(shakeBaseData, axis=0)
        pixelBleed = np.sum(pixelBleedData, axis=0)


        # Data Attributes
        nFrames = noise.shape[0]

        maxSignal = np.amax(shakeBase+noise+pixelBleed+shotNoise+structNoise+groundTruth)

        # Data Plotting
        imgs = []
        fig,axarr = plt.subplots(1,3)
        fig.suptitle("Point Source Tracking Data")
        for i in range(0,nFrames):
            base_img = shakeBase[i]+noise[i]+pixelBleed[i]+shotNoise[i]+structNoise[i]+groundTruth[i]
            detections_img = detections[i]
            struct_img = structNoise[i]
            axarr[0].imshow(base_img, vmin=0, vmax=maxSignal, cmap='Greys_r')
            axarr[1].imshow(detections_img, vmin=0, vmax=np.amax(detections_img), cmap='Greys_r')
            axarr[2].imshow(struct_img)

            for j in range(0, 2):
                axarr[j].set_xlim([0, 100])
                axarr[j].set_ylim([0, 100])
                axarr[j].set_xlabel("Distance East (km)")
                axarr[j].set_ylabel("Distance North (km)")

            axarr[0].set_title('ShakeBase with Noise, Pixel-Bleed Filters, and Shot Noise')
            axarr[1].set_title('Detections')

            plt.pause(0.05)

            for j in range(0, 2):
                axarr[j].cla()

def visualizeRegion_type2(region_name):
    # Read data
    with h5py.File(self.h5_path, 'r') as infile:
        group = infile[region_name]
        gt_subgroup = group["GroundTruth"]
        sb_subgroup = group["ShakeBase"]
        pb_subgroup = group["PixelBleed"]

        # Core subgroups
        groundTruthData = np.array([np.array(val) for val in gt_subgroup.values()])
        shakeBaseData = np.array([np.array(val) for val in sb_subgroup.values()])
        pixelBleedData = np.array([np.array(val) for val in pb_subgroup.values()])

        # Noise Layers
        noise = np.array(group["Noise"])
        structNoise = np.array(group['StructuredNoise'])
        shotNoise = np.array(group['ShotgunNoise'])

    # Consolidate gt, sb, and pb
    groundTruth = np.sum(groundTruthData, axis=0)
    shakeBase = np.sum(shakeBaseData, axis=0)
    pixelBleed = np.sum(pixelBleedData, axis=0)


    # Data Attributes
    nFrames, nRows, nCols = groundTruth.shape
    maxSignal = np.amax(shakeBase+noise+pixelBleed+structNoise)


    # Data Plotting
    imgs = []
    fig,axarr = plt.subplots(1,4)
    for i in range(nFrames):
        axarr[0].imshow(groundTruth[i], vmin=0, vmax=np.amax(groundTruth), cmap='Greys_r')
        axarr[1].imshow(shakeBase[i]+noise[i], vmin=0, vmax=np.amax(shakeBase+noise), cmap='Greys_r')
        axarr[2].imshow(shakeBase[i]+noise[i]+pixelBleed[i], vmin=0, vmax=np.amax(shakeBase+noise+pixelBleed), cmap='Greys_r')
        axarr[3].imshow(shakeBase[i]+noise[i]+pixelBleed[i]+structNoise[i]+shotNoise[i], vmin=0, vmax=maxSignal, cmap='Greys_r')

        for j in range(0, 4):
            axarr[j].set_xlim([0,100])
            axarr[j].set_ylim([0,100])
            axarr[j].set_xlabel("Distance East (km)")
            axarr[j].set_ylabel("Distance North (km)")

        fig.suptitle("Point Source Tracking Data")
        axarr[0].set_title('Ground Truth')
        axarr[1].set_title('ShakeBase with Noise Filter')
        axarr[2].set_title('ShakeBase with Noise and Pixel-Bleed Filters')
        axarr[3].set_title('ShakeBase with Structured Noise and Pixed-Bleed Filters')

        plt.pause(0.005)

        for j in range(0, 3):
            axarr[j].cla()

def loadingBar(loaded, total_i):

    remaining = (total_i-1) - loaded

    # Get percentage remaining
    percent = loaded / (total_i*1.0 - 1) * 100

    # Scale bar
    bar_scale = 0.35
    bar_length = int(100 * bar_scale) - 1
    loaded_scaled = int(loaded * bar_scale)
    remaining_scaled = bar_length - loaded_scaled


    # Write output to std out stream
    sys.stdout.write("\t[{}{}] {:0.0f}%".format("="*loaded_scaled, "-"*(remaining_scaled), percent))
    sys.stdout.write("\r")

    if remaining == 0:
        print("\n")
        return
    sys.stdout.flush()


def module_tester():
    pass

if __name__ == '__main__':
    module_tester()
