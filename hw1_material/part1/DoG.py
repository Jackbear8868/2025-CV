import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        h,w = image.shape[:2]
        kernel = (0,0)
        gaussian_images = []
        resized_image = cv2.resize(image, (w//2, h//2))

        gaussian_images.append(image)
        for i in range(self.num_DoG_images_per_octave):
            gaussian_images.append(cv2.GaussianBlur(image, kernel, self.sigma**(i+1)))

        gaussian_images.append(resized_image)
        for i in range(self.num_DoG_images_per_octave):
            gaussian_images.append(cv2.GaussianBlur(resized_image, kernel, self.sigma**(i+1)))
        

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves * self.num_guassian_images_per_octave):
            if i % self.num_guassian_images_per_octave == 0:
                continue
            tmp = cv2.subtract(gaussian_images[i],gaussian_images[i-1])
            dog_images.append(tmp)


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        # thresholded_images = []
        # for i in range(len(dog_images)):
        #     _, output = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
        #     thresholded_images.append(output)

        # dilated = cv2.dilate(image, np.ones((3, 3), np.uint8))  # Get max in local region
        # local_max = (image == dilated)  # Compare to original image
        # local_max[image < self.threshold] = 0  # Apply threshold

        # eroded = cv2.erode(image, np.ones((3, 3), np.uint8))  # Get min in local region
        # local_min = (image == eroded)  # Compare to original image
        # local_min[image > 255 - self.threshold] = 0  # Apply inverse threshold


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        
        keypoints = np.zeros(10,10)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints


if __name__ == '__main__':
    Gaussian = Difference_of_Gaussian(128)
    path = "testdata/1.png"
    image = cv2.imread(path)
    Gaussian.get_keypoints(image)