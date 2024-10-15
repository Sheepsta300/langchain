from __future__ import annotations

import logging
from typing import Any, Optional

from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from libs.core.langchain_core.callbacks.manager import CallbackManagerForToolRun
from libs.core.langchain_core.utils.env import get_from_dict_or_env
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class AzureTranslateTool(BaseModel):
    """
    A tool that interacts with the Azure Translator API using the SDK.

    This tool queries the Azure Translator API to translate text between
    languages. It requires an API key and endpoint, which can be set up as
    described in the Azure Translator API documentation:
    https://learn.microsoft.com/en-us/azure/ai-services/translator/
    translator-text-apis?tabs=python
    """

    text_translation_key: str
    text_translation_endpoint: str
    region: str
    translate_client: Optional[Any] = None  # Make Optional

    default_language: str = "en"
    name: str = "azure_translator_tool"
    description: str = (
        "A wrapper around Azure Translator API. Useful for translating text between "
        "languages. Input must be text (str). Ensure to install the "
        "azure-ai-translation-text package."
    )

    @field_validator(
        "text_translation_key", "text_translation_endpoint", "region", mode="before"
    )
    def validate_environment(cls, v, field):
        """
        Validate that the required environment variables are set.
        """
        value = get_from_dict_or_env({}, field.name, f"AZURE_{field.name.upper()}")
        if not value:
            raise ValueError(f"{field.name} is missing in environment variables")
        return value

    def setup_client(self):
        """Sets up the translation client."""
        if not self.translate_client:
            try:
                self.translate_client = TextTranslationClient(
                    endpoint=self.text_translation_endpoint,
                    credential=AzureKeyCredential(self.text_translation_key),
                    region=self.region,
                )
            except Exception as e:
                logger.error("Failed to set up the translation client: %s", e)
                raise

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
        self.setup_client()

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
