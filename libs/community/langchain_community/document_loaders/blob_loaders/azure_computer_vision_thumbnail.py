from __future__ import annotations

import os
from typing import Iterable, Optional

from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader
from langchain_community.tools.azure_ai_services.utils import (
    detect_file_src_type,
)


class AzureComputerVisionThumbnail(BlobLoader):
    def __init__(
        self,
        *,
        computer_vision_key: Optional[str] = None,
        computer_vision_endpoint: Optional[str] = None,
        thumbnail_name: Optional[str] = "thumbnail",
        image: str,
        width: int,
        height: int,
        save_dir: str,
    ):
        self.computer_vision_key = (
            computer_vision_key or os.environ["COMPUTER_VISION_KEY"]
        )
        self.computer_vision_endpoint = (
            computer_vision_endpoint or os.environ["COMPUTER_VISION_ENDPOINT"]
        )
        try:
            from azure.cognitiveservices.vision.computervision import (
                ComputerVisionClient,
            )
            from msrest.authentication import CognitiveServicesCredentials
        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-vision-computervision is not installed. "
                """Run `pip install azure-cognitiveservices-vision-computervision` 
                to install. """
            )

        self.computer_vision_client = ComputerVisionClient(
            endpoint=self.computer_vision_endpoint,
            credentials=CognitiveServicesCredentials(self.computer_vision_key),
        )
        self.image = image
        self.width = width
        self.height = height
        self.save_dir = save_dir
        if not thumbnail_name.endswith(".jpg"):
            self.thumbnail_name = f"{thumbnail_name}.jpg"
        else:
            self.thumbnail_name = thumbnail_name

    def yield_blobs(self) -> Iterable[Blob]:
        image = self._generate_thumbnail()
        bytes = b"".join(image)

        file_path = os.path.join(self.save_dir, self.thumbnail_name)
        with open(file_path, "wb") as image_file:
            image_file.write(bytes)

        loader = FileSystemBlobLoader(self.save_dir, glob="*.jpg")
        for blob in loader.yield_blobs():
            yield blob

    def _generate_thumbnail(self):
        image_src_type = detect_file_src_type(self.image)
        if image_src_type == "local":
            image_data = open(self.image, "rb")
            result = self.computer_vision_client.generate_thumbnail_in_stream(
                image=image_data, width=self.width, height=self.height
            )
        elif image_src_type == "remote":
            result = self.computer_vision_client.generate_thumbnail(
                url=self.image, width=self.width, height=self.height
            )
        else:
            raise ValueError(f"Invalid image path: {self.image}")

        return result
