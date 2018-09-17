"""Shrinks and augments downloaded images in order to create a training set."""
from __future__ import print_function, division
import os
import re
import random
import numpy as np
from scipy import misc, ndimage
from ImageAugmenter import ImageAugmenter

random.seed(42)

KEYWORDS = ["cloud", "clouds", "sky"]

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
READ_MAIN_DIR = os.path.join(MAIN_DIR, "downloaded/")
#READ_MAIN_DIR = "/media/aj/grab/ml/datasets/flickr-sky"
WRITE_MAIN_DIR = os.path.join(MAIN_DIR, "preprocessed/")
DIRS = [(os.path.join(READ_MAIN_DIR, "%s/" % (keyword,)), \
         os.path.join(WRITE_MAIN_DIR, "%s/" % (keyword,))) for keyword in KEYWORDS]

SCALE_HEIGHT = 256
SCALE_WIDTH = 256
RATIO_WIDTH_TO_HEIGHT = 1
EPSILON = 0.1

AUGMENTATIONS = 10
PADDING = 20

def main():
    """Iterates over the images in each directory, shrinks and augments each one."""
    nb_processed = 0
    nb_errors = 0
    nb_total = len(get_all_filepaths([download_dir for download_dir, write_to_dir in DIRS]))

    # iterate over directories (read-directory and save-to-directory)
    for download_dir, write_to_dir in DIRS:
        print("Reading from '%s'" % (download_dir,))
        print("Writing to '%s'" % (write_to_dir,))

        # create directory if it doesnt exist
        if not os.path.exists(write_to_dir):
            os.makedirs(write_to_dir)

        # load filepaths of images in directory
        fps_img = get_all_filepaths([download_dir])

        # iterate over each image
        for fp_img in fps_img:
            print("Image %d of %d (%.2f%%) (%s)" \
                  % (nb_processed+1, nb_total, 100*(nb_processed+1)/nb_total, fp_img))
            try:
                filename = fp_img[fp_img.rfind("/")+1:]

                # dont use misc.imload, fails for grayscale images
                image = ndimage.imread(fp_img, mode="RGB")
                image_orig = np.copy(image)
                #misc.imshow(image)
                #print(image)
                #print(image.shape)

                height = image_orig.shape[0]
                width = image_orig.shape[1]
                wh_ratio = width / height

                # add padding at the borders of the image
                # then augment image
                batch = np.zeros((AUGMENTATIONS, height+(2*PADDING), width+(2*PADDING), 3),
                                 dtype=np.uint8)

                img_padded = np.pad(image, ((PADDING, PADDING), (PADDING, PADDING), (0, 0)),
                                    mode="median")
                for i in range(0, AUGMENTATIONS):
                    batch[i] = np.copy(img_padded)

                ia = ImageAugmenter(width+(2*PADDING), height+(2*PADDING),
                                    channel_is_first_axis=False,
                                    hflip=True, vflip=False,
                                    scale_to_percent=(1.05, 1.2), scale_axis_equally=True,
                                    rotation_deg=5, shear_deg=1,
                                    translation_x_px=15, translation_y_px=15)
                batch = ia.augment_batch(batch)

                for i in range(0, AUGMENTATIONS):
                    # remove padding
                    image = batch[i, PADDING:-PADDING, PADDING:-PADDING, ...]

                    # shrink the image to desired height/width sizes
                    # first delete rows/columns until aspect ratio matches desired aspect ratio
                    # then resize
                    # doing this after the augmentation should decrease the likelihood of
                    # ending with badly looking black areas at the borders of the image
                    removed = 0
                    while not (wh_ratio - EPSILON <= RATIO_WIDTH_TO_HEIGHT <= wh_ratio + EPSILON):
                        if wh_ratio < RATIO_WIDTH_TO_HEIGHT:
                            # height value is too high
                            # remove more from top than from bottom, because we have sky images and
                            # hence much similar content at top and only a few rows of pixels with
                            # different content at the bottom
                            if removed % 4 != 0:
                                # remove one row at the top
                                image = image[1:height-0, :, ...]
                            else:
                                # remove one row at the bottom
                                image = image[0:height-1, :, ...]
                        else:
                            # width value is too high
                            if removed % 2 == 0:
                                # remove one column at the left
                                image = image[:, 1:width-0, ...]
                            else:
                                # remove one column at the right
                                image = image[:, 0:width-1, ...]

                        height = image.shape[0]
                        width = image.shape[1]
                        wh_ratio = width / height
                        removed += 1

                    image_resized = misc.imresize(image, (SCALE_HEIGHT, SCALE_WIDTH))

                    # save augmented image
                    filename_aug = filename.replace(".jp", "__%d.jp" % (i))
                    misc.imsave(os.path.join(write_to_dir, filename_aug), image_resized)
            except IOError as exc:
                # sometimes downloaded images cannot be read by imread()
                # this should catch these cases
                print("I/O error({0}): {1}".format(exc.errno, exc.strerror))
                nb_errors += 1

            nb_processed += 1

        print("Processed %d images" % (nb_processed,))
        print("Encountered %d errors" % (nb_errors,))
        print("Finished.")

def get_all_filepaths(fp_dirs):
    """Reads all filepaths to images in provided directories.
    Args:
        fp_dirs The list of directories
    Returns:
        List of filepaths to images
    """
    result_img = []
    for fp_dir in fp_dirs:
        fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
        fps = [os.path.join(fp_dir, f) for f in fps]
        fps_img = [fp for fp in fps if re.match(r".*\.(?:jpg|jpeg|png)$", fp)]
        result_img.extend(fps_img)

    return result_img

if __name__ == "__main__":
    main()
