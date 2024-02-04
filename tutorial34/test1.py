#write python program to extract images from pdf file using unstructured.partition.pdf module

from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import os 



OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

rpe = partition_pdf(
        filename="c.pdf",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=os.getcwd() + "/images"
    )


