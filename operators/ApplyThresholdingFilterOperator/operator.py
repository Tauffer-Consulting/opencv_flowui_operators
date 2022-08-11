from flowui.base_operator import BaseOperator
from .models import InputModel, OutputModel
from pathlib import Path
import cv2

def apply_binary(img):
    return cv2.threshold(img,127,255,cv2.THRESH_BINARY)

def apply_binary_inv(img):
    return cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

def apply_trunc(img):
    return cv2.threshold(img,127,255,cv2.THRESH_TRUNC)

def apply_tozero(img):
    return cv2.threshold(img,127,255,cv2.THRESH_TOZERO)

def apply_tozero_inv(img):
    return cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

def apply_adaptive(img):
    blur = cv2.medianBlur(img,5)
    return cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

def apply_adaptivemean(img):
    blur = cv2.medianBlur(img,5)
    return cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

def apply_otsus(img):
    return cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def apply_otsusgaussian(img):
    blur = cv2.medianBlur(img,5)
    return cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

class ApplyThresholdingFilter(BaseOperator):

    def operator_function(self, input_model: InputModel):
        #Read the image
        image = cv2.imread(input_model.input_file_path)

        effect_types_map = dict(
            binary = apply_binary,
            binary_inv = apply_binary_inv,
            trunc = apply_trunc,
            tozero = apply_tozero,
            tozero_inv = apply_tozero_inv,
            adaptive = apply_adaptive,
            adaptivemean = apply_adaptivemean,
            otsus = apply_otsus,
            otsusgaussian = apply_otsusgaussian
        )

        chosen_effect = input_model.effect

        image_processed = effect_types_map[chosen_effect](img=image)

        # Save result
        out_file_path = str(Path(self.results_path) / Path(input_model.input_file_path).name)
        cv2.imwrite(out_file_path, image_processed)

        return OutputModel(
            message=f"Filtered image successfully saved to: {out_file_path}",
            output_file_path=str(out_file_path)
        )
