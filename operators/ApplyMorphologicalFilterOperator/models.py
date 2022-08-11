from pydantic import BaseModel, Field
from enum import Enum

class EffectType(str, Enum):
    erosion = "erosion"
    dilation = "dilation"
    opening = "opening"
    closing = "closing"
    morphologicalgradient = "morphologicalgradient"
    tophat = "tophat"
    blackhat = "blackhat"

class InputModel(BaseModel):
    """
    Apply effect to image
    """

    input_file_path: str = Field(
        description="Path to the input file"
    )
    effect: EffectType = Field(
        default='erosion',
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
