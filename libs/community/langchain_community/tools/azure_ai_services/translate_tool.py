from __future__ import annotations

import logging
import os
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class AzureTranslateTool(BaseTool):
    """
    A tool that interacts with the Azure Translator API using the SDK.

    This tool queries the Azure Translator API to translate text between
    languages. It requires an API key and endpoint, which can be set up as
    described in the Azure Translator API documentation:
    https://learn.microsoft.com/en-us/azure/ai-services/translator/
    translator-text-apis?tabs=python
    """

    translate_client: Any = None
    default_language: str = "en"

    # New class attributes
    text_translation_key: str = ""
    text_translation_endpoint: str = ""
    region: str = ""

    name: str = "azure_translator_tool"
    description: str = (
        "A wrapper around Azure Translator API. Useful for translating text between "
        "languages. Input must be text (str). Ensure to install the "
        "azure-ai-translation-text package."
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.validate_environment()

    def validate_environment(self):
        """
        Validate that the required environment variables are set, and set up
        the client.
        """
        try:
            from azure.ai.translation.text import TextTranslationClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure-ai-translation-text is not installed. "
                "Run `pip install azure-ai-translation-text` to install."
            )

        # Get environment variables
        self.text_translation_key = os.getenv("AZURE_TRANSLATE_API_KEY")
        self.text_translation_endpoint = os.getenv("AZURE_TRANSLATE_ENDPOINT")
        self.region = os.getenv("REGION")

        if not self.text_translation_key:
            raise ValueError(
                "AZURE_TRANSLATE_API_KEY is missing in environment variables"
            )
        if not self.text_translation_endpoint:
            raise ValueError(
                "AZURE_TRANSLATE_ENDPOINT is missing in environment variables"
            )
        if not self.region:
            raise ValueError("REGION is missing in environment variables")

        # Set up the translation client
        self.translate_client = TextTranslationClient(
            endpoint=self.text_translation_endpoint,
            credential=AzureKeyCredential(self.text_translation_key),
            region=self.region,
        )

    def _translate_text(self, text: str, to_language: str = "en") -> str:
        """
        Perform text translation using the Azure Translator API.

        Args:
            text (str): The text to be translated.
            to_language (str): The target language to translate to.

        Returns:
            str: The translation result.
        """
        if not text:
            raise ValueError("Input text for translation is empty.")

        # Ensure that the translation client is initialized
        if not self.translate_client:
            self.validate_environment()

        body = [{"Text": text}]
        try:
            response = self.translate_client.translate(
                body=body, to_language=[to_language]
            )
            return response[0].translations[0].text
        except Exception as e:
            logger.error("Translation failed: %s", e)
            raise RuntimeError(f"Translation failed: {e}")

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        to_language: str = "en",
    ) -> str:
        """
        Run the tool to perform translation.

        Args:
            query (str): The text to be translated.
            run_manager (Optional[CallbackManagerForToolRun]):
             A callback manager for tracking the tool run.
            to_language (str): The target language for translation.

        Returns:
            str: The translated text.
        """
        return self._translate_text(query, to_language)
