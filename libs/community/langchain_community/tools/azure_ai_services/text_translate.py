from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

logger = logging.getLogger(__name__)


class AzureAITextTranslateTool(BaseTool):
    """
    A tool that interacts with the Azure Translator API using the SDK.

    Attributes:
        text_translation_key (str):
         The API key for the Azure Translator service.
        text_translation_endpoint (str):
        The endpoint for the Azure Translator service.
        region (str): The Azure region where the Translator service is hosted.
        translate_client (TextTranslationClient):
        The Azure Translator client initialized with the credentials.
        default_language (str):
        The default language for translation (default is 'en').

    This tool queries the Azure Translator API to translate text between languages.
    Input must be text (str), and the 'to_language'
     parameter must be a two-letter language code (e.g., 'es', 'it', 'de').
    """

    text_translation_key: Optional[str] = None
    text_translation_endpoint: Optional[str] = None
    region: Optional[str] = None

    translate_client: Any = None
    default_language: str = "en"

    name: str = "azure_translator_tool"
    description: str = (
        "A wrapper around Azure Translator API. Useful for translating text between "
        "languages. Input must be text (str), and the 'to_language'"
        " parameter must be a two-letter language code (str), e.g., 'es', 'it', 'de'."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """
        Validate that the required environment variables are set, and set up
        the client.
        """

        try:
            from azure.ai.translation.text import TextTranslationClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "Azure Translator API packages are not installed. "
                "Please install 'azure-ai-translation-text' and 'azure-core'."
            )

        text_translation_key = get_from_dict_or_env(
            values, "text_translation_key", "AZURE_TRANSLATE_API_KEY"
        )
        text_translation_endpoint = get_from_dict_or_env(
            values, "text_translation_endpoint", "AZURE_TRANSLATE_ENDPOINT"
        )
        region = get_from_dict_or_env(values, "region", "REGION")

        values["translate_client"] = TextTranslationClient(
            endpoint=text_translation_endpoint,
            credential=AzureKeyCredential(text_translation_key),
            region=region,
        )

        return values

    def _translate_text(self, text: str, to_language: str = "en") -> str:
        """
        Perform text translation using the Azure Translator API.
        """
        if not text:
            raise ValueError("Input text for translation is empty.")

        body = [{"Text": text}]
        try:
            response = self.translate_client.translate(
                body=body, to_language=[to_language]
            )
            return response[0].translations[0].text
        except Exception as e:
            logger.error("Translation failed: %s", e)
            raise RuntimeError(f"Translation failed: {e}")

        from azure.ai.translation.text import TextTranslationClient

        self.translate_client: TextTranslationClient

        body = [{"Text": text}]
        try:
            response = self.translate_client.translate(
                body=body, to_language=[to_language]
            )
            return response[0].translations[0].text
        except Exception:
            raise

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
        # Ensure only the text (not the full query dictionary)
        # is passed to the translation function
        text_to_translate = query
        return self._translate_text(text_to_translate, to_language)