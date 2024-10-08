from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

logger = logging.getLogger(__name__)


class AzureAIFaceAnalysisTool(BaseTool):
    """
    A wrapper for the Azure AI Face API.

    This tool facilitates interaction with the Azure AI Face Analysis API,
    enabling developers to analyze images for facial features and
    recognition attributes. The analysis can be performed on local image files,
    with configurable attributes and models for both detection and recognition.

    Attributes:
        azure_ai_face_key (Optional[str]):
            The API key for Azure AI Face Analysis.
        azure_ai_face_endpoint (Optional[str]):
            The endpoint URL for Azure AI Face Analysis.
        detection_attributes (Optional[List[str]]):
            Attributes to be used during face detection.
        recognition_attributes (Optional[List[str]]):
            Attributes to be used during face recognition.
        detection_model (Any):
            The detection model to use (default: 'DETECTION_03').
        recognition_model (Any):
            The recognition model to use (default: 'RECOGNITION_04').
        formatted_attributes (Any):
            Formatted attributes to be used during analysis.
        face_id_time_to_live (int):
            Time to live for face ID (default: 120 seconds).
        return_recognition_model (bool):
            Whether to return recognition model details (default: True).
        return_face_landmarks (bool):
            Whether to return face landmarks (default: True).
        return_face_id (bool):
            Whether to return face IDs (default: True).
    """

    azure_ai_face_key: Optional[str] = None  #: :meta private:
    azure_ai_face_endpoint: Optional[str] = None  #: :meta private:
    face_client: Any = None  #: :meta private:
    detection_attributes: Optional[List[str]] = ["HEAD_POSE"]
    recognition_attributes: Optional[List[str]] = ["QUALITY_FOR_RECOGNITION"]
    detection_model: Any = "DETECTION_03"
    recognition_model: Any = "RECOGNITION_04"
    formatted_attributes: Any = None  #: :meta private:
    face_id_time_to_live: int = 120
    return_recognition_model: bool = True
    return_face_landmarks: bool = True
    return_face_id: bool = True

    name: str = "azure_ai_face_analysis"
    description: str = (
        "A wrapper around Azure AI Face API. "
        "Useful for when you need to analyze images with potential faces/people"
        "Input should be a local path to an image."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        azure_ai_face_key = get_from_dict_or_env(
            values, "azure_ai_face_key", "AZURE_AI_FACE_KEY"
        )

        azure_ai_face_endpoint = get_from_dict_or_env(
            values, "azure_ai_face_endpoint", "AZURE_AI_FACE_ENDPOINT"
        )

        """Validate that azure.ai.vision.face is installed."""
        try:
            from azure.ai.vision.face import FaceClient
            from azure.ai.vision.face.models import (
                FaceAttributeTypeDetection03,
                FaceAttributeTypeRecognition04,
                FaceDetectionModel,
                FaceRecognitionModel,
            )
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure.ai.vision.face is not installed. "
                "Run `pip install azure.ai.vision.face` to install. "
            )

        """Validate Azure AI Vision Face Analysis client can be initialized."""
        try:
            values["face_client"] = FaceClient(
                endpoint=azure_ai_face_endpoint,
                credential=AzureKeyCredential(azure_ai_face_key),
            )
        except Exception as e:
            raise RuntimeError(
                f"Initialization of Azure AI Face Analysis client failed: {e}"
            )

        attributes = []
        for att in values.get("detection_attributes", ["HEAD_POSE"]):
            attributes.append(FaceAttributeTypeDetection03[att])
        for att in values.get("recognition_attributes", ["QUALITY_FOR_RECOGNITION"]):
            attributes.append(FaceAttributeTypeRecognition04[att])

        rec_model = values.get("recognition_model", "RECOGNITION_04")
        det_model = values.get("recognition_model", "DETECTION_03")
        values["recognition_model"] = FaceRecognitionModel[rec_model]
        values["detection_model"] = FaceDetectionModel[det_model]

        values["formatted_attributes"] = attributes

        values["face_id_time_to_live"] = values.get("face_id_time_to_live", 120)
        values["return_recognition_model"] = values.get(
            "return_recognition_model", True
        )
        values["return_face_landmarks"] = values.get("return_face_landmarks", True)
        values["return_face_id"] = values.get("return_face_id", True)

        return values

    def _face_analysis(self, image_path: str) -> List:
        """
        Analyze an image using Azure AI Face Analysis.

        This method performs facial detection and recognition on the given
        image using the configured attributes, detection model, and recognition model.

        Args:
            image_path (str): Path to the local image.

        Returns:
            List: A list containing the results of the face analysis,
            including detected attributes.
        """

        from azure.ai.vision.face import FaceClient
        from azure.ai.vision.face.models import (
            FaceDetectionModel,
            FaceRecognitionModel,
        )

        self.face_client: FaceClient
        self.formatted_attributes: List
        self.detection_model: FaceDetectionModel
        self.recognition_model: FaceRecognitionModel

        with open(image_path, "rb") as fd:
            file_content = fd.read()

        result = self.face_client.detect(
            file_content,
            detection_model=self.detection_model,  # The latest detection model.
            recognition_model=self.recognition_model,  # The latest recognition model.
            return_face_attributes=self.formatted_attributes,
            return_face_id=self.return_face_id,
            return_face_landmarks=self.return_face_landmarks,
            return_recognition_model=self.return_recognition_model,
            face_id_time_to_live=self.face_id_time_to_live,
        )

        return result

    def _format_face_analysis_result(self, result: List) -> str:
        """
        Format the results from the face analysis into a readable string.

        This method takes the analysis results and formats them into a human-readable
        string that includes information such as face attributes and
        recognition details.

        Args:
            result (List): The results of the face analysis.

        Returns:
            str: A formatted string containing the face analysis results. If no
            faces are detected, returns "No faces found".
        """
        if not result:
            return "No faces found"

        formatted: str = ""
        for idx, face in enumerate(result):
            formatted += f"FACE: {idx+1}\n"
            for k, v in face.items():
                formatted += f"{k}: {v}\n"
        return formatted

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute the face analysis tool.

        This method runs the face analysis based on the provided query and returns
        a formatted result.

        Args:
            query (str): A local path to the image to be analyzed.
            run_manager (Optional[CallbackManagerForToolRun]):
                Optional callback manager for managing the run.

        Returns:
            str: The formatted face analysis result.

        Raises:
            RuntimeError: If an error occurs while running the analysis.
        """
        try:
            image_analysis_result = self._face_analysis(query)
            return self._format_face_analysis_result(image_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running AzureAIFaceAnalysisTool: {e}")
