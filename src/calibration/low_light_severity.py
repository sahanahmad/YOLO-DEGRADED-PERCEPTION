import cv2
import numpy as np
class LowLightSeverity:
    def __init__(self):
        self.p33 = None
        self.p66 = None

    def score(self, image_bgr: np.ndarray)-> float:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        return float(np.mean(v_channel))

    def fit(self, luminance_values):
        self.p33 = np.percentile(luminance_values, 33)
        self.p66 = np.percentile(luminance_values,66)

    def severity_bin(self, score):
        if self.p33 is None or self.p66 is None:
            raise ValueError ("Must call fit() before severty_bin()")
        if score < self.p33:
            return 'high' #low light  
        elif score < self.p66:
            return 'medium' # medium light
        else:
            return 'low' #high light- the image is ok