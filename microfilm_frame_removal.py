import cv2
import sys
import os

def removeMicroFrame(image_path, output_folder='./cropped/'):
	image = cv2.imread(image_path)
	crop_img = image[1300:, 0:]
	basename = os.path.basename(image_path)
	cv2.imwrite(output_folder + basename, crop_img)

if __name__ == '__main__':
	input_folder = './crf_green/'
	for file in os.listdir(input_folder):
		print('> processing:', file)
		removeMicroFrame(input_folder + file)
