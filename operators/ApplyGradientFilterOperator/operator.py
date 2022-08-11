from flowui.base_operator import BaseOperator
from .models import InputModel, OutputModel

from pathlib import Path
import cv2

def apply_laplacian(img):
    return cv2.Laplacian(img,cv2.CV_64F)

def apply_sobelx(img):
    return cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

def apply_sobely(img):
    return cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)



class ApplyGradientFilter(BaseOperator):

    def operator_function(self, input_model: InputModel):
        #Read the image
        image = cv2.imread(input_model.input_file_path)
        
        effect_types_map = dict(
            laplacian = apply_laplacian,
            sobelx = apply_sobelx,
            sobely = apply_sobely,
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
