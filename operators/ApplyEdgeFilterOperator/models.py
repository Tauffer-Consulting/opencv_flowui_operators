from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    Apply effect to image
    """

    input_file_path: str = Field(
        description="Path to the input file"
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
