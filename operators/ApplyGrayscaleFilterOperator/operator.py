from flowui.base_operator import BaseOperator
from .models import InputModel, OutputModel

from pathlib import Path
import cv2


class ApplyGrayscaleFilter(BaseOperator):

    def operator_function(self, input_model: InputModel):
        #Read the image
        image = cv2.imread(input_model.input_file_path)

        image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save result
        out_file_path = str(Path(self.results_path) / Path(input_model.input_file_path).name)
        cv2.imwrite(out_file_path, image_processed)

        return OutputModel(
            message=f"Filtered image successfully saved to: {out_file_path}",
            output_file_path=str(out_file_path)
        )
