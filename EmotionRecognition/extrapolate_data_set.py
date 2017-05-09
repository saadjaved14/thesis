import glob
import helper
import cv2
import os

# Load YAML setup file
gd_setup, gd_config = helper.load_setup()

# Get the location of data set to extrapolate
gv_dataset_path = gd_setup['sourceFilesPath']


def apply_gaussian_pyramid(pv_path, repeat_count, rescale = False):
    # read all the files from the given path
    ll_image_files = glob.glob(os.path.join(pv_path, '*', '*'))

    # iterate through all images
    # ignore files that are not images (*.png, *.jpg)
    for img_path in ll_image_files:
        # Split image filename and extension
        img_filename, img_ext = os.path.splitext(img_path)

        if img_ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
            # read image in to a cv object
            img = cv2.imread(img_path)

            for i in xrange(repeat_count):
                # pyramid down the image
                img_pyr = cv2.pyrDown(img)
                # store the pyramid down image
                cv2.imwrite(img_filename + '_ex_' + str(i) + img_ext, img_pyr)
                # swap images for repeat
                img = img_pyr


if __name__ == '__main__':
    apply_gaussian_pyramid(gv_dataset_path, 2)