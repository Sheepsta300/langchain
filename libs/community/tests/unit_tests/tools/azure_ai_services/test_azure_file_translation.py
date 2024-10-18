from pathlib import Path
from typing import Any

import pytest

from langchain_community.tools.azure_ai_services.azure_file_translation import (
    AzureFileTranslateTool,
)

_THIS_DIR = Path(__file__).parents[3]

_EXAMPLES_DIR = _THIS_DIR / "examples"
AZURE_PDF = _EXAMPLES_DIR / "test_azure.pdf"


@pytest.mark.requires("azure.ai.translation.text")
@pytest.mark.requires("unstructured")
@pytest.mark.requires("pi_heif")
def test_tool_initialization(mocker: Any) -> None:
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)

    mock_translate_client = mocker.Mock()
    mocker.patch(
        "azure.ai.translation.text.TextTranslationClient",
        return_value=mock_translate_client,
    )

    key = "key"
    endpoint = "endpoint"
    region = "westus2"

    tool = AzureFileTranslateTool(
        text_translation_key=key,
        text_translation_endpoint=endpoint,
        region=region,
        translate_client=mock_translate_client,
    )

    assert tool.text_translation_key == key
    assert tool.text_translation_endpoint == endpoint
    assert tool.region == region
    assert tool.translate_client == mock_translate_client


@pytest.mark.requires("azure.ai.translation.text")
@pytest.mark.requires("unstructured")
@pytest.mark.requires("pi_heif")
def test_translation_with_file(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"
    region = "westus2"

    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)

    mock_translate_client = mocker.Mock()
    mocker.patch(
        "azure.ai.translation.text.TextTranslationClient",
        return_value=mock_translate_client,
    )

    tool = AzureFileTranslateTool(
        text_translation_key=key,
        text_translation_endpoint=endpoint,
        region=region,
        translate_client=mock_translate_client,
    )

    mock_translate_client.translate.return_value = [
        {
            "detectedLanguage": {"language": "en", "score": 1.0},
            "translations": [{"text": "Hola, mi nombre es Azure", "to": "es"}],
        }
    ]

    file_input: str = str(AZURE_PDF)
    expected_output = "Hola, mi nombre es Azure"

    result = tool._run(file_input)

    assert result == expected_output


@pytest.mark.requires("azure.ai.translation.text")
@pytest.mark.requires("unstructured")
@pytest.mark.requires("pi_heif")
def test_translation_with_no_file(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"
    region = "westus2"

    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)

    mock_translate_client = mocker.Mock()
    mocker.patch(
        "azure.ai.translation.text.TextTranslationClient",
        return_value=mock_translate_client,
    )

    tool = AzureFileTranslateTool(
        text_translation_key=key,
        text_translation_endpoint=endpoint,
        region=region,
        translate_client=mock_translate_client,
    )

    file_input: str = ""
    expected_output = "Error while running AzureFileTranslateTool"

    try:
        result = tool._run(file_input)
    except RuntimeError as e:
        result = str(e)

    assert expected_output in result
