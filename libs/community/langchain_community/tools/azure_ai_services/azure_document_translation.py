from __future__ import annotations

import os
import logging
from typing import Optional, Any

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

logger = logging.getLogger(__name__)


class AzureDocumentTranslateTool(BaseTool):
    """
    A tool that uses Azure Text Translation API to translate a text document from
    any language into a target language.
    """

    text_translation_key: str = ""
    text_translation_endpoint: str = ""
    target_language: str = "es"  # Default target language set to Spanish
    client: Any

    name: str = "azure_document_translation"
    description: str = (
        """
        A Wrapper around Azure AI Services can be used to
        translate a document into a specific language.
        It reads the text from a file, processes it,
        and then outputs with the desired language.
        """
    )

    def __init__(self, *,
                 text_translation_key: Optional[str] = None,
                 text_translation_endpoint: Optional[str] = None
                 ) -> None:
        super().__init__()

        try:
            from azure.ai.translation.text import TextTranslationClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "The Azure Text Translation SDK is not installed. "
                "Run `pip install azure-ai-translation-text` to install."
            )

        self.text_translation_key = (
                text_translation_key or os.getenv("TEXT_TRANSLATION_KEY")
        )
        self.text_translation_endpoint = (
                text_translation_endpoint or os.getenv("TEXT_TRANSLATION_ENDPOINT")
        )

        if not all([self.text_translation_key, self.text_translation_endpoint]):
            raise ValueError(
                "Azure Cognitive Services key and endpoint must be provided"
            )

        # Initialize client after ensuring imports are successful
        self.client = TextTranslationClient(
            endpoint=self.text_translation_endpoint,
            credential=AzureKeyCredential(self.text_translation_key)
        )

    def read_text_from_file(self, file_path: str) -> str:
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

        try:
            from langchain_community.document_loaders import (
                UnstructuredPowerPointLoader,
                UnstructuredWordDocumentLoader,
                UnstructuredPDFLoader,
                UnstructuredExcelLoader,
                UnstructuredXMLLoader,
                UnstructuredHTMLLoader)
        except ImportError:
            raise ImportError(
                """
                The Langchain Unstructured Document loader is not installed. 
                First run `pip install langchain-unstructured` to install.
                
                To install the dependencies for all document types, 
                use 'pip install "unstructured[all-docs]"'
                """
            )

        file_extension = os.path.splitext(file_path)[1].lower()

        # Map file extensions to loader classes
        loader_map = {
            ".pdf": UnstructuredPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xml": UnstructuredXMLLoader,
            ".html": UnstructuredHTMLLoader
        }

        loader_class = loader_map.get(file_extension)

        if file_extension == ".txt":
            # Handle plain text files directly
            return self.read_text(file_path)
        elif loader_class is None:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Load the document using the appropriate loader
        loader = loader_class(file_path)
        data = loader.load()

        return " ".join([doc.page_content for doc in data])

    def read_text(self, file_path: str) -> str:
        """Read text from a plain text file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def translate_text(self, text: str, target_language: Optional[str] = None) -> str:
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
            raise ImportError(
                "Run 'pip install azure-ai-translation-text'.")

        try:
            request_body = [InputTextItem(text=text)]
            response = self.client.translate(content=request_body, to=[target_language])

            translations = response[0].translations
            if translations:
                return translations[0].text
            return ""  # No translations found
        except Exception as e:
            logger.error(f"Error occurred during translation: {e}")
            raise

    def _run(self, query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """"Run the tool"""
        try:
            return self.translate_text(query)
        except Exception as e:
            logger.error(f"Error while running AzureDocumentTranslateTool: {e}")
            raise RuntimeError(
                f"Error while running AzureDocumentTranslateTool: {e}")

