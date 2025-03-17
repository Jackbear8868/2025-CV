import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def calculate_Y(self, image, wr, wg, wb):
    # Extract R, G, B channels
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)

    # Compute Y channel
    Y = wr * R + wg * G + wb * B

    return Y.astype(np.uint8)  # Convert back to uint8 for image processing



def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    wr, wg, wb = 0.5, 0.3, 0.2
    Y = calculate_Y()

if __name__ == '__main__':
    main()