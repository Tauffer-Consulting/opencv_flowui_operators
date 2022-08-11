from flowui.base_operator import BaseOperator
from .models import InputModel, OutputModel
from pathlib import Path
import cv2


# greyscale filter
def apply_averaging(img):
    return cv2.blur(img,(5,5))

def apply_gaussianblurring(img):
    return cv2.GaussianBlur(img,(5,5),0)

def apply_medianblurring(img):
    return cv2.medianBlur(img,5)

def apply_bilateralfiltering(img):
    return cv2.bilateralFilter(img,9,75,75)

class ApplyBlurFilter(BaseOperator):

    def operator_function(self, input_model: InputModel):
        #Read the image
        image = cv2.imread(input_model.input_file_path)

        effect_types_map = dict(
            averaging = apply_averaging,
            gaussianblurring = apply_gaussianblurring,
            medianblurring = apply_medianblurring,
            bilateralfiltering = apply_bilateralfiltering,
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
