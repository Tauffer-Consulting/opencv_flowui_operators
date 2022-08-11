from flowui.base_operator import BaseOperator
from .models import InputModel, OutputModel
from pathlib import Path
import cv2
import numpy as np


def apply_erosion(img, kernel):
    return cv2.erode(img,kernel,iterations = 1)

def apply_dilation(img, kernel):
    return cv2.dilate(img,kernel,iterations = 1)

def apply_opening(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def apply_closing(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def apply_morphologicalgradient(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def apply_tophat(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def apply_blackhat(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

class ApplyMorphologicalFilter(BaseOperator):

    def operator_function(self, input_model: InputModel):
        #Read the image
        image = cv2.imread(input_model.input_file_path)
        kernel = np.ones((5,5),np.uint8)

        effect_types_map = dict(
            erosion = apply_erosion,
            dilation = apply_dilation,
            opening = apply_opening,
            closing = apply_closing,
            morphologicalgradient = apply_morphologicalgradient,
            tophat = apply_tophat,
            blackhat = apply_blackhat
        )
        chosen_effect = input_model.effect
        image_processed = effect_types_map[chosen_effect](img=image, kernel=kernel)

        # Save result
        out_file_path = str(Path(self.results_path) / Path(input_model.input_file_path).name)
        cv2.imwrite(out_file_path, image_processed)

        return OutputModel(
            message=f"Filtered image successfully saved to: {out_file_path}",
            output_file_path=str(out_file_path)
        )
