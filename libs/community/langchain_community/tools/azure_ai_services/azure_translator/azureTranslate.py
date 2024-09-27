from __future__ import annotations

import logging
import os
from typing import Any, Optional

from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

# Setup logging
logging.basicConfig(level=logging.WARNING)
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
    translate_client: Any = None  #: :meta private:

    name: str = "azure_translator_tool"
    description: str = (
        "A wrapper around Azure Translator API. Useful for translating text between "
        "languages. Input must be text (str). Ensure to install the azure-ai-translation-"
        "text package."
    )

    def __init__(
        self, *, translate_key: Optional[str] = None, 
        translate_endpoint: Optional[str] = None
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
            translate_key=translate_key,
            translate_endpoint=translate_endpoint
        )

        self.translate_client = TextTranslationClient(
            endpoint=translate_endpoint,
            credential=AzureKeyCredential(translate_key)
        )

    def _translate_text(self, text: str, to_language: str) -> str:
        """
        Perform text translation using the Azure Translator API.

        Args:
            text (str): The text to be translated.
            to_language (str): The target language to translate to.

        Returns:
            str: The translation result.
        """
        if not text:
            logger.error("Input text for translation is empty.")
            return None

        body = [{"Text": text}]
        try:
            response = self.translate_client.translate(
                body=body,
                to_language=[to_language]
            )
            if response:
                logger.warning(
                    f"Translation successful: {response[0].translations[0].text}"
                )  # Use WARNING level for successful operations
                return response[0].translations[0].text
            else:
                logger.error("Translation failed with an empty response")
                return None
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    def _run(
        self, query: str, to_language: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
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
        try:
            return self._translate_text(query, to_language)
        except Exception as e:
            raise RuntimeError(f"Error while running AzureTranslateTool: {e}")

    @classmethod
    def from_env(cls):
        """
        Create an instance of the tool using environment variables.
        """
        translate_key = os.getenv("AZURE_OPENAI_TRANSLATE_API_KEY")
        translate_endpoint = os.getenv("AZURE_OPENAI_TRANSLATE_ENDPOINT")

        if not translate_key:
            raise ValueError("AZURE_TRANSLATE_API_KEY is missing in environment variables")
        if not translate_endpoint:
            raise ValueError("AZURE_TRANSLATE_ENDPOINT is missing in environment variables")

        logger.info(f"API Key: {translate_key[:4]}**** (masked)")
        logger.info(f"Endpoint: {translate_endpoint}")

        return cls(translate_key=translate_key, translate_endpoint=translate_endpoint)


# Example test usage for the AzureTranslateTool
if __name__ == "__main__":
    tool = AzureTranslateTool.from_env()
    try:
        translated_text = tool._run("good morning, How are you?", 'es')
        logger.info(f"Translated text: {translated_text}")
    except RuntimeError as e:
        logger.error(f"Error occurred: {e}")
