from __future__ import annotations

import logging
import tempfile
from typing import Dict

from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class AzureOpenAIText2SpeechTool(BaseTool):
    """Tool that queries the Azure OpenAI Text2Speech API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/ai-services/openai/text-to-speech-quickstart?tabs=command-line
    """

    azure_openai_key: str = ""
    openai_endpoint: str = ""
    speech_voice: str = "alloy" 
    response_format: str = "mp3"  
    speech_speed: str = "1"

    name: str = "azure_openai_text2speech"
    description: str = (
        "A wrapper around Azure OpenAI Text2Speech. "
        "Useful for when you need to convert text to speech. "
    )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        if values is None:
            raise ValueError("No values were provided to the validator")

        azure_openai_key = get_from_dict_or_env(
            values, "azure_openai_key", "AZURE_OPENAI_API_KEY"
        )

        openai_endpoint = get_from_dict_or_env(
            values, "openai_endpoint", "AZURE_OPENAI_ENDPOINT"
        )

        try:
            import httpx

        except ImportError:
            raise ImportError(
                "httpx is not installed. "
                "Run `pip install httpx` to install."
            )
        
        return values
    

    def _text2speech(self, text: str, speech_voice: str, response_format: str, speech_speed: str) -> str:
        
        url = f"{self.openai_endpoint}/openai/deployments/tts/audio/speech?api-version=2024-02-15-preview"
        headers = {
            "api-key": self.azure_openai_key,
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1-hd",
            "input": text,
            "voice": speech_voice,
            "response_format" : response_format,
            "speed" : speech_speed
        }
        
        try:
            import httpx
            response = httpx.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an error for HTTP error responses

            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=f".{response_format}", delete=False
                ) as f:
                    f.write(response.content)
                return f.name
            else:
                return f"Speech synthesis failed with status code {response.status_code}: {response.text}"
        
        except httpx.RequestError as e:
            logger.error(f"An error occurred while making the request: {e}")
            return f"Request failed: {e}"
        
    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        try:
            speech_file = self._text2speech(query, speech_voice=self.speech_voice, response_format=self.response_format, speech_speed=self.speech_speed)
            return speech_file
        except Exception as e:
            raise RuntimeError(f"Error while running AzureOpenAIText2SpeechTool: {e}")