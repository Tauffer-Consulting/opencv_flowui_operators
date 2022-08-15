from flowui.base_operator import BaseOperator
from .models import InputModel, OutputModel
from pathlib import Path
import cv2
from skimage import io
import base64
import numpy as np


# greyscale filter
def apply_averaging(img):
    return cv2.blur(img,(5,5))


def apply_gaussianblurring(img):
    return cv2.GaussianBlur(img,(5,5),0)


def apply_medianblurring(img):
    return cv2.medianBlur(img,5)


def apply_bilateralfiltering(img):
    return cv2.bilateralFilter(img,9,75,75)


def apply_base64_image(img):
    _, encoded_img = cv2.imencode('.png', img)  # Works for '.jpg' as well
    base64_img = base64.b64encode(encoded_img).decode("utf-8")
    return base64_img


def apply_numpy_array_image(img):
    return img


class ApplyBlurFilter(BaseOperator):

    def operator_function(self, input_model: InputModel):
        #Read the image
        if input_model.direct_link:
            nparray = io.imread(input_model.direct_link)
            image = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)
        elif input_model.input_file_path:
            image = cv2.imread(input_model.input_file_path)
        elif input_model.base64_image:
            encoded_data = input_model.base64_image.split(',')[1]
            nparray = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            image = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)
        elif input_model.numpy_array:
            image = cv2.imdecode(input_model.numpy_array, cv2.IMREAD_UNCHANGED)
        else:
            return OutputModel(
                message="ERROR: No image provided",
                output_file_path=None
            )

        effect_types_map = dict(
            averaging = apply_averaging,
            gaussianblurring = apply_gaussianblurring,
            medianblurring = apply_medianblurring,
            bilateralfiltering = apply_bilateralfiltering,
        )

        chosen_effect = input_model.effect

        image_processed = effect_types_map[chosen_effect](img=image)

        # Save result
        if input_model.image_format:
            format_map = dict(
            base64_image = apply_base64_image,
            numpy_array_image = apply_numpy_array_image
        )
            chosen_format = input_model.image_format
            image_data = format_map[chosen_format](img=image_processed)
            return OutputModel(
                message="Image successfully downloaded and sent through XCOM.",
                output_file_path=None,
                image_data = image_data
            )
        else:
            out_file_path = str(Path(self.results_path) / Path(input_model.input_file_path).name)
            cv2.imwrite(out_file_path, image_processed)

            return OutputModel(
                message=f"Filtered image successfully saved to: {out_file_path}",
                output_file_path=str(out_file_path)
            )
