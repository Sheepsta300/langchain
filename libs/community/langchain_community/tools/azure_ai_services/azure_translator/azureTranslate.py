from __future__ import annotations

import logging
import os
from typing import Any, Optional, Dict
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

class AzureTranslateTool(BaseTool):
    """
    A tool that interacts with the Azure Translator API using the SDK.

    This tool queries the Azure Translator API to translate text between languages.
    It requires an API key and endpoint, which can be set up as described in the
    Azure Translator API documentation. https://learn.microsoft.com/en-us/azure/ai-services/translator/translator-text-apis?tabs=python
    """

    translate_key: str = ""
    translate_endpoint: str = ""
    translate_client: Any = None  #: :meta private:

    name: str = "azure_translator_tool"
    description: str = (
        "A wrapper around Azure Translator API. "
        "Useful for translating text between languages. Input must be text (str)."
        """must have pip install azure-ai-translation-text"""
    )

    def __init__(self, *, translate_key: Optional[str] = None, translate_endpoint: Optional[str] = None) -> None:
        """
        Initialize the AzureTranslateTool with the given API key and endpoint.
        """
        translate_key = translate_key or os.environ.get("AZURE_OPENAI_TRANSLATE_API_KEY")
        translate_endpoint = translate_endpoint or os.environ.get("AZURE_OPENAI_TRANSLATE_ENDPOINT")

        if not translate_key or not translate_endpoint:
            raise ValueError("Missing API key or endpoint for Azure Translator API.")

        # Initialize parent class (Pydantic)
        super().__init__(
            translate_key=translate_key,
            translate_endpoint=translate_endpoint
        )

        # Initialize the Translator Client outside of Pydantic attributes
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
        # Check for empty input and raise a ValueError
        if not text:
            raise ValueError("Input text for translation is empty.")

        # The request body should contain a list of dictionaries, where each dictionary contains the text to be translated
        body = [{"Text": text}]  # Use "Text" as the key in the body (based on Translator API)

        try:
            # Correct call to the SDK, ensuring that the body and to_language are passed properly
            response = self.translate_client.translate(
                body=body,  # The body should be passed here
                to_language=[to_language]  # The target language must be passed as a list
            )

            if response:
                # Extract and return the translation result
                return response[0].translations[0].text
            else:
                raise ValueError("Translation failed with an empty response.")
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Error during translation: {e}")

    def _run(self, query: str, to_language: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Run the tool to perform translation.

        Args:
            query (str): The text to be translated.
            to_language (str): The target language to translate to. No default value.
            run_manager (Optional[CallbackManagerForToolRun], optional): A callback manager for tracking the tool run.

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

        print(f"API Key: {translate_key[:4]}**** (masked)")
        print(f"Endpoint: {translate_endpoint}")

        return cls(translate_key=translate_key, translate_endpoint=translate_endpoint)


# Example test usage for the AzureTranslateTool
if __name__ == "__main__":
    # Set up your environment variables or pass the API key and endpoint directly
    tool = AzureTranslateTool.from_env()

    # Test translating
    try:
        translated_text = tool._run("good morning, Stephen and Amjed", 'es')
        print(f"Translated text: {translated_text}")
    except RuntimeError as e:
        print(f"Error occurred: {e}")
