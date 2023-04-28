'''
This file can be used to test the models on input images. It includes a feature for CRF postprocessing.

For pixel-wise separation of real documents always add --enableCRF

Run python 'classifier_fcnn.py -h' for more information
'''

import argparse
import sys
import warnings
import os.path
import numpy as np
import skimage.io as io
from fcn_helper_function import weighted_categorical_crossentropy, IoU
from img_utils import getbinim, max_rgb_filter, mask2rgb, rgb2mask
from keras.engine.saving import load_model
from post import crf
from skimage import img_as_float
from skimage.color import gray2rgb

if not sys.warnoptions:
    warnings.simplefilter("ignore")

BOXWDITH = 256
STRIDE = BOXWDITH - 10

def classify(image):
    model = load_model('./models/scanned.h5', custom_objects={'loss': weighted_categorical_crossentropy([0.4,0.5,0.1]), 'IoU': IoU})
    orgim = np.copy(image)
    image = img_as_float(gray2rgb(getbinim(image)))
    maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
    maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
    mask = np.ones((maskh, maskw, 3))
    mask2 = np.zeros((maskh, maskw, 3))
    mask[0:image.shape[0], 0:image.shape[1]] = image
    print("classifying image...")
    for y in range(0, mask.shape[0], STRIDE):
        x = 0
        if (y + BOXWDITH > mask.shape[0]):
            break
        while (x + BOXWDITH) < mask.shape[1]:
            input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
            std = input.std() if input.std() != 0 else 1
            mean = input.mean()
            mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                    np.array([(input-mean)/std]))[0]
            x = x + STRIDE
    return mask2[0:image.shape[0], 0:image.shape[1]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enableCRF", help="Use crf for postprocessing",
                        action="store_true")
    parser.add_argument('-i', "--input_image", help="input image file name")
    parser.add_argument('-o', "--output_folder", help="output folder")
    args = parser.parse_args()
    inputim = io.imread(args.input_image)
    output_folder = args.output_folder
    args = parser.parse_args()
    title = os.path.basename(args.input_image)[:-4]
    subfolders = ["out", "out_crf"]
    for s in subfolders:
        if not os.path.exists(s):
    	    os.mkdir(s)
    out = classify(inputim)

    if args.enableCRF:
        crf_res = crf(inputim, out)
    else:
        crf_res = None

    if crf_res is not None:
            io.imsave(output_folder + '/out/' + title + '_out.png', max_rgb_filter(out))
            print('saved fcn_out.png')
            io.imsave(output_folder + '/out_crf/' + title + '_out_crf.png', mask2rgb(crf_res))
            print('saved fcn_out_post.png')
    else:
            io.imsave(output_folder + '/out/' + title + '_out.png', max_rgb_filter(out))
            print('saved fcn_out.png')
