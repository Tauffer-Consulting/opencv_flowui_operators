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


class InputModel(BaseModel):
    """
    Apply effect to image
    """

    input_file_path: str = Field(
        description="Path to the input file"
    )
    effect: EffectType = Field(
        default='random',
        description='Effect to be applied'
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