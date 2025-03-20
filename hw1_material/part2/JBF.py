
import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
        offset = np.arange(-self.pad_w, self.pad_w + 1, dtype=np.float64)
        x, y = np.meshgrid(offset, offset)
        self.G_s = np.exp(-(x**2 + y**2) / (2 * self.sigma_s**2))

    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        padded_guidance = padded_guidance / 255.0
        h, w = img.shape[:2]
        output = np.zeros_like(img, dtype=np.float64)

        # Precompute spatial kernel (fixed for the entire image)

        for i in range(h):
            for j in range(w):
                i_p, j_p = i + self.pad_w, j + self.pad_w # Adjusted index in padded image

                # Extract local window
                img_window = padded_img[i_p - self.pad_w:i_p + self.pad_w + 1, j_p - self.pad_w:j_p + self.pad_w + 1]
                guidance_window = padded_guidance[i_p - self.pad_w:i_p + self.pad_w + 1, j_p - self.pad_w:j_p + self.pad_w + 1]
                
                # Compute range kernel G_r using guidance image
                if guidance.ndim == 2:  # Grayscale image
                    G_r = np.exp(-((guidance_window - padded_guidance[i_p, j_p])**2) / (2 * self.sigma_r**2))
                else:  # RGB image
                    diff_r = (guidance_window[:, :, 0] - padded_guidance[i_p, j_p, 0])**2
                    diff_g = (guidance_window[:, :, 1] - padded_guidance[i_p, j_p, 1])**2
                    diff_b = (guidance_window[:, :, 2] - padded_guidance[i_p, j_p, 2])**2
                    G_r = np.exp(-(diff_r + diff_g + diff_b) / (2 * self.sigma_r**2))
                
                # Compute final joint bilateral weights
                weights = self.G_s * G_r

                if img.ndim == 2:  # Grayscale image
                    output[i, j] = np.sum(weights * img_window) / np.sum(weights)
                else:  # RGB image (process each channel separately)
                    output[i, j] = np.sum(weights[..., np.newaxis] * img_window, axis=(0, 1)) / np.sum(weights)

        return np.clip(output, 0, 255).astype(np.uint8)