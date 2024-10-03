from __future__ import annotations

import logging
import os
from typing import Any, Optional

from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class AzureTranslateTool(BaseTool):
    """
    A tool that interacts with the Azure Translator API using the SDK.

    This tool queries the Azure Translator API to translate text between languages.
    It requires an API key and endpoint, which can be set up as described in the
    Azure Translator API documentation:
    https://learn.microsoft.com/en-us/azure/ai-services/translator/
    translator-text-apis?tabs=python
    """

    translate_key: str = ""
    translate_endpoint: str = ""
    translate_client: Any = None
    default_language: str = "en"

    name: str = "azure_translator_tool"
    description: str = (
        "A wrapper around Azure Translator API. Useful for translating text between "
        "languages. Input must be text (str). Ensure to install the "
        "azure-ai-translation-text package."
    )

    @classmethod
    def validate_environment(cls):
        """
        Validate that the required environment variables are set.
        """
        translate_key = os.getenv("AZURE_OPENAI_TRANSLATE_API_KEY")
        translate_endpoint = os.getenv("AZURE_OPENAI_TRANSLATE_ENDPOINT")

        if not translate_key:
            raise ValueError("AZURE_TRANSLATE_API_KEY is missing in environment variables")

        if not translate_endpoint:
            raise ValueError("AZURE_TRANSLATE_ENDPOINT is missing in environment variables")

        return cls(translate_key=translate_key, translate_endpoint=translate_endpoint)

    def __init__(
        self,
        *,
        translate_key: Optional[str] = None,
        translate_endpoint: Optional[str] = None,
    ) -> None:
        """
        Initialize the AzureTranslateTool with the given API key and endpoint.
        """
        translate_key = translate_key or os.getenv("AZURE_OPENAI_TRANSLATE_API_KEY")
        translate_endpoint = translate_endpoint or os.getenv(
            "AZURE_OPENAI_TRANSLATE_ENDPOINT"
        )

        if not translate_key or not translate_endpoint:
            raise ValueError("Missing API key or endpoint for Azure Translator API.")

        super().__init__(
            translate_key=translate_key, translate_endpoint=translate_endpoint
        )

        self.translate_client = TextTranslationClient(
            endpoint=translate_endpoint, credential=AzureKeyCredential(translate_key)
        )

    def _translate_text(self, text: str, to_language: Optional[str] = None) -> str:
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

        body = [{"Text": text}]
        to_language = to_language or self.default_language

        try:
            response = self.translate_client.translate(
                body=body, to_language=[to_language]
            )
            if response:
                return response[0].translations[0].text
            else:
                raise RuntimeError("Translation failed with an empty response")
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}")

    def _run(
        self,
        query: str,
        to_language: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Run the tool to perform translation.

        Args:
            query (str): The text to be translated.
            to_language (str): The target language to translate to.
            run_manager (Optional[CallbackManagerForToolRun]): A callback manager
            for tracking the tool run.

        Returns:
            str: The translated text.
        """
        return self._translate_text(query, to_language)
