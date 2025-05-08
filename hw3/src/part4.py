import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    threshold = 3
    max_iter = 5000
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        # Match descriptors.
        matches = bf.match(des1,des2)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
    
        # TODO: 2. apply RANSAC to choose best H
        max_inliers = []
        best_H = None

        for i in range(max_iter):
            subset = random.sample(matches, 4)
            u = np.array([kp2[m.trainIdx].pt for m in subset])
            v = np.array([kp1[m.queryIdx].pt for m in subset])

            H = solve_homography(u, v)
            if H is None:
                continue

            inliers = []
            for m in matches:
                pt1 = np.array([*kp1[m.queryIdx].pt, 1.0])
                pt2 = np.array([*kp2[m.trainIdx].pt, 1.0])

                projected = H @ pt2
                projected /= projected[2]

                error = np.linalg.norm(projected[:2] - pt1[:2])
                if error < threshold:
                    inliers.append(m)

            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_H = H

        u_best = np.array([kp2[m.trainIdx].pt for m in max_inliers])
        v_best = np.array([kp1[m.queryIdx].pt for m in max_inliers])
        best_H = solve_homography(u_best, v_best)

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H

        # TODO: 4. apply warping
        h, w = dst.shape[:2]
        dst = warping(im2, dst, last_best_H, 0, h, 0, w, direction='b')
        out = dst
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)