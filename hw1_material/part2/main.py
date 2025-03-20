import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def calculate_Y(image, wr, wg, wb):
    # Extract R, G, B channels
    R = image[:, :, 0].astype(np.float64)
    G = image[:, :, 1].astype(np.float64)
    B = image[:, :, 2].astype(np.float64)

    # Compute Y channel
    Y = wr * R + wg * G + wb * B

    return Y.astype(np.float64)  # Convert back to uint8 for image processing

def compute_L1_norm(image1, image2):
    """ 計算 L1-norm (絕對誤差總和) """
    return np.sum(np.abs(image1.astype(np.int64) - image2.astype(np.int64)))


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output/gray.png", img_gray)

    ### TODO ###
    with open(args.setting_path, "r") as file:
        file.readline()
        RGBs = []
        for _ in range(5):
            RGBs += [tuple(float(num) for num in file.readline().strip().split(','))]
        tmp = file.readline().strip().split(',')
        sigma_s,sigma_r = int(tmp[1]),float(tmp[3])

    jbf = Joint_bilateral_filter(sigma_s,sigma_r)
    
    # 1️⃣ **用原始彩圖作為 guidance 進行 Joint Bilateral Filtering**
    filtered_rgb = jbf.joint_bilateral_filter(img_rgb.astype(np.int64), img_rgb.astype(np.int64))
    cv2.imwrite("output/filtered_rgb.png", cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR))

    # 2️⃣ **用不同參數的灰階影像作為 guidance**
    grayscale_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]  # OpenCV 預設灰階
    grayscale_images += [calculate_Y(img_rgb, RGB[0], RGB[1], RGB[2]) for RGB in RGBs]

    best_cost = float('inf')
    worst_cost = float('-inf')
    best_gray_index, worst_gray_index = -1, -1
    filtered_best, filtered_worst = None, None
    gray_best, gray_worst = None, None

    # 3️⃣ **Apply JBF with grayscale images as guidance and compute L1-norm**
    for i, gray_img in enumerate(grayscale_images):
        filtered_gray_guided_rgb = jbf.joint_bilateral_filter(img_rgb.astype(np.int64), gray_img.astype(np.int64))

        # Compute L1 Norm
        L1_norm = compute_L1_norm(filtered_rgb, filtered_gray_guided_rgb)
        print(f"Grayscale Image {i}: L1 Norm = {L1_norm}")

        # Store best and worst results
        if L1_norm < best_cost:
            best_cost = L1_norm
            best_gray_index = i
            filtered_best = filtered_gray_guided_rgb
            gray_best = gray_img

        if L1_norm > worst_cost:
            worst_cost = L1_norm
            worst_gray_index = i
            filtered_worst = filtered_gray_guided_rgb
            gray_worst = gray_img

    # 4️⃣ **Save best & worst grayscale and their filtered results**
    cv2.imwrite("output/filtered_lowest_cost.png", cv2.cvtColor(filtered_best, cv2.COLOR_RGB2BGR))
    cv2.imwrite("output/filtered_highest_cost.png", cv2.cvtColor(filtered_worst, cv2.COLOR_RGB2BGR))
    cv2.imwrite("output/gray_lowest_cost.png", gray_best)
    cv2.imwrite("output/gray_highest_cost.png", gray_worst)

    print(f"Best grayscale image index: {best_gray_index} with L1 Norm = {best_cost}")
    print(f"Worst grayscale image index: {worst_gray_index} with L1 Norm = {worst_cost}")

    
if __name__ == '__main__':
    main()