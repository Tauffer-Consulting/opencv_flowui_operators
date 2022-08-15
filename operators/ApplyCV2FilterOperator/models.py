from pydantic import BaseModel, Field
from enum import Enum

class EffectType(str, Enum):
    random = "random"
    grayscale = "grayscale"
    bright = "bright"
    dark = "dark"
    sharp = "sharp"
    sepia = "sepia"
    pencil = "pencil"
    pencil_color = "pencil_color"
    hdr = "hdr"
    invert = "invert"
    summer = "summer"
    winter = "winter"


class ImageFormat(str, Enum):
    numpy_array = 'narray'
    base64 = 'base64'


class InputModel(BaseModel):
    """
    Apply effect to image
    """

    input_file_path: str = Field(
        default=None,
        description="Path to the input file"
    )
    direct_link: str = Field(
        default=None,
        description="Direct url for image / Convert image to numpy array"
    )
    base64_image: str = Field(
        default=None,
        description="Base64 image"
    )
    numpy_array: str = Field(
        default=None,
        description="Numpy Array image"
    )
    effect: EffectType = Field(
        default='random',
        description='Effect to be applied'
    )
    format: ImageFormat = Field(
        default=None,
        description='Image format'
    )


class OutputModel(BaseModel):
    """
    Apply effect to image
    """

    message: str = Field(
        default="",
        description="Output message to log"
    )
    output_file_path: str = Field(
        description="Path to the output file"
    )
    image_data: str = Field(
        default=None,
        description="Return image data as numpy array or base64"
    )
