import numpy as np
import cv2
import argparse

def compute_L1_norm(image1, image2):
    """ Compute the L1-norm (absolute difference sum) between two images """
    if image1.shape != image2.shape:
        raise ValueError(f"Image dimensions do not match! {image1.shape} vs {image2.shape}")
    
    return np.sum(np.abs(image1.astype(np.int32) - image2.astype(np.int32)))

def main():
    parser = argparse.ArgumentParser(description="Compute L1-Norm cost between two images")
    parser.add_argument("--image1", required=True, help="Path to the first image")
    parser.add_argument("--image2", required=True, help="Path to the second image")
    args = parser.parse_args()

    # Read images
    img1 = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img2 = cv2.imread(args.image2, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    if img1 is None or img2 is None:
        raise ValueError("One or both images could not be loaded. Check file paths!")

    # Compute L1-Norm
    L1_cost = compute_L1_norm(img1, img2)
    print(f"L1-Norm Cost between images: {L1_cost}")

if __name__ == "__main__":
    main()
