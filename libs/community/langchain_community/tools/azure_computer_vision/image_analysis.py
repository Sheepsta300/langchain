from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.tools.azure_ai_services.utils import (
    detect_file_src_type,
)

logger = logging.getLogger(__name__)


class AzureCogsComputerVisionTool(BaseTool):
    """Tool that queries the Azure Cognitive Services Computer Vision API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/python/api/overview/azure/cognitiveservices-vision-computervision-readme?view=azure-python-previous#get-text-description-of-an-image
    """

    computer_vision_key: Optional[str] = None  #: :meta private:
    computer_vision_endpoint: Optional[str] = None  #: :meta private:
    computer_vision_client: Any  #: :meta private:
    visual_features: list[str] = ["tags"]  #: :meta private:

    name: str = "azure_cogs_computer_vision_tool"
    description: str = (
        "A wrapper around Azure Cognitive Services Computer Vision. "
        "Useful for when you need to analyze images. "
        "Input must be a url to an image, or a local file path."
    )

    def __init__(
        self,
        *,
        computer_vision_key: Optional[str] = None,
        computer_vision_endpoint: Optional[str] = None,
        visual_features: list[str] = ["tags"],
    ):
        computer_vision_key = computer_vision_key or os.environ["COMPUTER_VISION_KEY"]
        computer_vision_endpoint = (
            computer_vision_endpoint or os.environ["COMPUTER_VISION_ENDPOINT"]
        )
        try:
            from azure.cognitiveservices.vision.computervision import (
                ComputerVisionClient,
            )
            from azure.cognitiveservices.vision.computervision.models import (
                VisualFeatureTypes,
            )
            from msrest.authentication import CognitiveServicesCredentials
        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-vision-computervision is not installed. "
                """Run `pip install azure-cognitiveservices-vision-computervision` 
                to install. """
            )

        computer_vision_client = ComputerVisionClient(
            endpoint=computer_vision_endpoint,
            credentials=CognitiveServicesCredentials(computer_vision_key),
        )

        visual_features = [VisualFeatureTypes[feature] for feature in visual_features]
        super().__init__(
            computer_vision_key=computer_vision_key,
            computer_vision_endpoint=computer_vision_endpoint,
            computer_vision_client=computer_vision_client,
            visual_features=visual_features,
        )

    # @model_validator(mode="before")
    # @classmethod
    # def validate_environment(cls, values: Dict) -> Dict:
    #     """Validate that api key and endpoint exists in environment."""
    #     computer_vision_key = get_from_dict_or_env(
    #         values, "computer_vision_key", "COMPUTER_VISION_KEY"
    #     )

    #     computer_vision_endpoint = get_from_dict_or_env(
    #         values, "computer_vision_endpoint", "COMPUTER_VISION_ENDPOINT"
    #     )

    #     try:
    #         from azure.cognitiveservices.vision.computervision import (
    #                               ComputerVisionClient
    #                               )
    #         from azure.cognitiveservices.vision.computervision.models import (
    #                                               VisualFeatureTypes
    #                                               )
    #         from msrest.authentication import CognitiveServicesCredentials
    #     except ImportError:
    #         raise ImportError(
    #             "azure-cognitiveservices-vision-computervision is not installed. "
    #             '''Run `pip install azure-cognitiveservices-vision-computervision`
    #                to install. '''
    #         )

    #     try:
    #         values["computer_vision_client"] = ComputerVisionClient(
    #             endpoint=computer_vision_endpoint,
    #             credentials=CognitiveServicesCredentials(computer_vision_key)
    #         )
    #     except Exception as e:
    #         raise RuntimeError(
    #             f'''Initialization of Azure AI Vision
    #               Image Analysis client failed: {e}'''
    #         )

    #     values["visual_features"] = [feature for feature in VisualFeatureTypes]

    #     return values

    def _image_analysis(self, image_path: str) -> Dict:
        # try:
        #     from azure.cognitiveservices.vision.computervision import (
        #                                       ComputerVisionClient
        #                                       )
        #     from msrest.authentication import CognitiveServicesCredentials
        # except ImportError:
        #     pass

        # self.computer_vision_client = ComputerVisionClient(
        #     endpoint=self.endpoint,
        #     credentials=CognitiveServicesCredentials(self.computer_vision_key)
        # )

        # visual_features = (
        #       [VisualFeatureTypes[feature] for feature in self.visual_features]
        #       )

        image_src_type = detect_file_src_type(image_path)
        if image_src_type == "local":
            image_data = open(image_path, "rb")
            result = self.computer_vision_client.analyze_image_in_stream(
                image=image_data,
                visual_features=self.visual_features,
            )
        elif image_src_type == "remote":
            result = self.computer_vision_client.analyze_image(
                url=image_path,
                visual_features=self.visual_features,
            )
        else:
            raise ValueError(f"Invalid image path: {image_path}")

        extracted_info = {}
        if result.adult is not None:
            extracted_info["adult"] = result.adult
        if result.brands is not None:
            extracted_info["brands"] = result.brands
        if result.categories is not None:
            extracted_info["categories"] = result.categories
        if result.color is not None:
            extracted_info["color"] = result.color
        if result.description is not None:
            extracted_info["description"] = result.description
        if result.faces is not None:
            extracted_info["faces"] = result.faces
        if result.image_type is not None:
            extracted_info["image_type"] = result.image_type
        if result.objects is not None:
            extracted_info["objects"] = result.objects
        if result.tags is not None:
            extracted_info["tags"] = result.tags

        return extracted_info

    def _format_image_analysis_result(self, image_analysis_result: Dict) -> str:
        formatted_result = []
        if "adult" in image_analysis_result:
            formatted_result.append(
                f"""
                Contains adult content: 
                {image_analysis_result['adult'].is_adult_content}, 
                {image_analysis_result['adult'].adult_score}\n
                Contains racy content: 
                {image_analysis_result['adult'].is_racy_content}, 
                {image_analysis_result['adult'].racy_score}\n
                Contains gory content: 
                {image_analysis_result['adult'].is_gory_content}, 
                {image_analysis_result['adult'].gore_score}\n
                """
            )

        if (
            "brands" in image_analysis_result
            and len(image_analysis_result["brands"]) > 0
        ):
            formatted_result.append(
                "Brands: " + ", ".join(image_analysis_result["brands"])
            )

        if "categories" in image_analysis_result:
            categories = ""
            for c in image_analysis_result["categories"]:
                categories += f"{c.name} - Score : {c.score}\n"
            formatted_result.append(categories)

        if "colour" in image_analysis_result:
            formatted_result.append(f"""
               Dominant colours: {image_analysis_result['color'].dominant_colors}\n
               Foreground: {image_analysis_result['color'].dominant_color_foreground}\n
               Background: {image_analysis_result['color'].dominant_color_background}\n
               Accent colours: {image_analysis_result['color'].accent_colors}\n
               Is Black and White: {image_analysis_result['color'].is_bwg_image}\n
            """)

        if "description" in image_analysis_result:
            captions = ""
            for c in image_analysis_result["description"].captions:
                captions += f"{c.text} - Confidence : {c.confidence}\n"
            formatted_result.append(captions)

        if (
            "faces" in image_analysis_result and len(image_analysis_result["faces"])
        ) > 0:
            formatted_result.append(
                "Faces: " + ", ".join(image_analysis_result["faces"])
            )

        if "image_type" in image_analysis_result:
            formatted_result.append(f"""
               Line art type: {image_analysis_result['image_type'].line_drawing_type}\n
               Clip art type: {image_analysis_result['image_type'].clip_art_type}\n
               """)

        if "objects" in image_analysis_result:
            objects = ""
            for o in image_analysis_result["objects"]:
                objects += f"""   {o.object_property}
                                                    Confidence: {o.confidence}\n
                                                    Location: {o.rectangle}
                                                """
            formatted_result.append(objects)

        if "tags" in image_analysis_result:
            tags = ""
            for t in image_analysis_result["tags"]:
                tags += f"""{t.name}\n
                        Confidence : {t.confidence}\n"""
            formatted_result.append(tags)
        return "\n".join(formatted_result)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            image_analysis_result = self._image_analysis(query)

            return self._format_image_analysis_result(image_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running AzureCogsComputerVisionTool: {e}")
