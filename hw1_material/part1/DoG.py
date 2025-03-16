import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_3d_extrema(self, dog_images):
        keypoints = []  # Use list instead of np.array for efficient append operations

        for i in range(self.num_octaves):
            octave_dog = dog_images[i]
            for j in range(1, self.num_DoG_images_per_octave - 1):  # Ignore first & last DoG
                for y in range(1, octave_dog[j].shape[0] - 1):
                    for x in range(1, octave_dog[j].shape[1] - 1):
                        value = octave_dog[j][y, x]
                        if np.abs(value) < self.threshold:
                            continue  # Ignore weak keypoints
                        
                        # Extract 3x3x3 neighborhood
                        neighbors = np.concatenate([
                            np.array(octave_dog[j - 1][y-1:y+2, x-1:x+2]).flatten(),  # Below
                            np.array(octave_dog[j][y-1:y+2, x-1:x+2]).flatten(),      # Same
                            np.array(octave_dog[j + 1][y-1:y+2, x-1:x+2]).flatten()   # Above
                        ])
                        
                        if value >= np.max(neighbors) or value <= np.min(neighbors):
                            if i == 0:
                                keypoints.append((y, x))
                            else:
                                keypoints.append((y*2, x*2))

        return np.array(keypoints, dtype=np.int32)  # Convert to NumPy array at the end

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        h, w = image.shape[:2]
        kernel = (0,0)
        gaussian_images = []

        gaussian_images.append(image)
        for i in range(1,self.num_guassian_images_per_octave):
            gaussian_images.append(cv2.GaussianBlur(image, kernel, self.sigma ** i))

        resized_image = cv2.resize(gaussian_images[-1], (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        gaussian_images.append(resized_image)
        for i in range(1,self.num_guassian_images_per_octave):
            gaussian_images.append(cv2.GaussianBlur(resized_image, kernel, self.sigma ** i))

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)

        dog_images = []
        for i in range(self.num_octaves):
            dogs = []
            for j in range(self.num_DoG_images_per_octave):
                dog = cv2.subtract(gaussian_images[i * self.num_guassian_images_per_octave + j + 1], gaussian_images[i * self.num_guassian_images_per_octave + j])
                dogs.append(dog)
                cv2.imwrite(f"output/DoG-{i}-{j}.png",dog)
            dog_images.append(dogs)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        keypoints = self.get_3d_extrema(dog_images)

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints


if __name__ == '__main__':
    Gaussian = Difference_of_Gaussian(5)
    path = "testdata/1.png"
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints = Gaussian.get_keypoints(image)