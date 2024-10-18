import unittest
from unittest.mock import patch, MagicMock
from libs.community.langchain_community.tools.azure_ai_services.translate_tool import (
    AzureTranslateTool,
)


class TestAzureTranslateTool(unittest.TestCase):
    @patch(
        "azure.ai.translation.text.TextTranslationClient"
    )  # Mock the translation client
    def setUp(self, mock_translation_client):
        # Set up a mock translation client
        self.mock_translation_client = mock_translation_client
        self.mock_translation_instance = mock_translation_client.return_value

        # Create an instance of the AzureTranslateTool
        self.tool = AzureTranslateTool()

        # Mock environment variables
        self.tool.text_translation_key = "fake_api_key"
        self.tool.text_translation_endpoint = "https://fake.endpoint.com"
        self.tool.region = "fake_region"

    def test_translate_success(self):
        # Mock the translation API response
        mock_response = MagicMock()
        mock_response.translations = [MagicMock(text="Hola")]
        self.mock_translation_instance.translate.return_value = [mock_response]

        # Call the translate function
        result = self.tool._translate_text("Hello", to_language="es")

        # Assert the result is as expected
        self.assertEqual(result, "Hola")
        self.mock_translation_instance.translate.assert_called_once_with(
            body=[{"Text": "Hello"}], to_language=["es"]
        )

    def test_empty_text(self):
        # Test that an empty input raises a ValueError
        with self.assertRaises(ValueError) as context:
            self.tool._translate_text("", to_language="es")

        self.assertEqual(str(context.exception), "Input text for translation is empty.")

    def test_api_failure(self):
        # Simulate an API failure
        self.mock_translation_instance.translate.side_effect = Exception("API failure")

        # Test that an exception is raised and handled properly
        with self.assertRaises(RuntimeError) as context:
            self.tool._translate_text("Hello", to_language="es")

        self.assertIn("Translation failed", str(context.exception))

    @patch("azure.ai.translation.text.TextTranslationClient")
    def test_validate_environment(self, mock_translation_client):
        # Test that the environment is validated and the client is initialized
        self.tool.validate_environment()

        # Ensure the translation client is initialized properly
        mock_translation_client.assert_called_once_with(
            endpoint=self.tool.text_translation_endpoint, credential=unittest.mock.ANY
        )


# Run the unit tests
if __name__ == "__main__":
    unittest.main()
