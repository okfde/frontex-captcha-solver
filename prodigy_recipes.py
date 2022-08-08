# Based on https://github.com/explosion/prodigy-recipes/blob/master/image/image_caption/image_caption.py

import prodigy
from prodigy.components.loaders import Images
from pathlib import Path
from prodigy.components.filters import filter_duplicates
import base64
from prodigy import set_hashes

from typing import List, Dict
import prodigy
from prodigy.components.db import Database, Dataset, Link, Example


@prodigy.recipe("image-caption")
def image_caption(dataset, images_path):
    """
    Stream in images from a directory and allow captioning them by typing
    a caption in a text field. The caption is stored as the key "caption".
    """
    stream = Images(images_path)
    stream = [set_hashes(eg) for eg in stream]
    stream = filter_duplicates(stream, by_input=True, by_task=True)

    blocks = [
        {"view_id": "image"},
        {"view_id": "text_input", "field_id": "caption", "field_autofocus": True},
    ]
    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": {"blocks": blocks},
        # "exclude_by": "input",
    }


@prodigy.recipe("write-images")
def write_images(dataset: str, output: str):
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)

    DB: Database = prodigy.components.db.connect()
    if dataset not in DB:
        raise ValueError(f"Dataset {dataset} not found!")

    dataset_id = Dataset.get(Dataset.name == dataset).id
    links = Link.select(Link.example).where(Link.dataset == dataset_id)
    to_delete: List[Link] = []
    for link in links:
        content = link.example.load()
        caption = content["caption"]
        header, image_content = content["image"].split(",")
        content_type = header.split(":")[1].split(";")[0]
        extension = content_type.split("/")[1]
        with open(output_path / f"{caption}.{extension}", "wb") as f:
            f.write(base64.b64decode(image_content))
