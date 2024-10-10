from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env

from langchain_community.document_loaders import (
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredXMLLoader,
)

logger = logging.getLogger(__name__)


class AzureFileTranslateTool(BaseTool):
    """
    A tool that uses Azure Text Translation API to translate a text document from
    any language into a target language.
    """

    text_translation_key: str = ""
    text_translation_endpoint: str = ""
    target_language: str = "en"
    translate_client: Any

    name: str = "azure_document_translation"
    description: str = """
        A Wrapper around Azure AI Services can be used to
        translate a document into a specific language.
        It reads the text from a file, processes it,
        and then outputs with the desired language.
        """

    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """
        Validate that the API key and endpoint exist in the environment.
        """
        azure_translate_key = get_from_dict_or_env(
            values, "text_translation_key", "AZURE_TRANSLATE_API_KEY"
        )
        azure_translate_endpoint = get_from_dict_or_env(
            values, "text_translation_endpoint", "AZURE_TRANSLATE_ENDPOINT"
        )

        try:
            from azure.ai.translation.text import TextTranslationClient
            from azure.core.credentials import AzureKeyCredential

            # Set up the translation client in the values dict
            values["translate_client"] = TextTranslationClient(
                endpoint=azure_translate_endpoint,
                credential=AzureKeyCredential(azure_translate_key),
            )

        except ImportError:
            raise ImportError(
                "azure-ai-translation-text is not installed. "
                "Run `pip install azure-ai-translation-text` to install."
            )

        return values

    def _read_text_from_file(self, file_path: str) -> str:
        """
        Read and return text from the specified file,
        supporting PDF, DOCX, PPTX, XLSX, HTML, and XML formats.

        Args:
            file_path (str): Path to the input file.

        Returns:
            str: Extracted text from the file.

        Raises:
            ValueError: If the file type is unsupported.
        """

        file_extension = os.path.splitext(file_path)[1].lower()

        # Map file extensions to loader classes
        loader_map = {
            ".pdf": UnstructuredPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xml": UnstructuredXMLLoader,
            ".html": UnstructuredHTMLLoader,
        }

        loader_class = loader_map.get(file_extension)

        if file_extension == ".txt":
            # Handle plain text files directly
            return self._read_text(file_path)
        elif loader_class is None:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Load the document using the appropriate loader
        loader = loader_class(file_path)
        data = loader.load()

        return " ".join([doc.page_content for doc in data])

    def _read_text(self, file_path: str) -> str:
        """Read text from a plain text file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def _translate_text(self, text: str, target_language: Optional[str] = None) -> str:
        """
        Translate the input text to the target language
        using the Azure Text Translation API.

        Args:
            text (str): The text to be translated.
            target_language (str, optional):
                The target language for translation (default: Spanish).

        Returns:
            str: Translated text.

        Raises:
            RuntimeError: If the translation request fails.
        """
        if target_language is None:
            target_language = self.target_language

        try:
            from azure.ai.translation.text.models import InputTextItem
        except ImportError:
            raise ImportError("Run 'pip install azure-ai-translation-text'.")

        try:
            request_body = [InputTextItem(text=text)]
            response = self.translate_client.translate(
                content=request_body, to=[target_language]
            )

            translations = response[0].translations
            if translations:
                return translations[0].text
            return ""  # No translations found
        except Exception as e:
            raise RuntimeError(f"An error occurred during translation: {e}")

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """ "Run the tool"""
        try:
            return self._translate_text(query)
        except Exception as e:
            raise RuntimeError(f"Error while running AzureFileTranslateTool: {e}")
