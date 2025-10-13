"""
FalAI Model implementation for smolagents.

This module provides a custom Model class that wraps fal.ai's API
to work with smolagents' CodeAgent and other agent types.
"""

import logging
from typing import Any, Generator

import fal_client
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    Model,
    get_clean_message_list,
)
from smolagents.monitoring import TokenUsage
from smolagents.tools import Tool


logger = logging.getLogger(__name__)


class FalAIModel(Model):
    """
    A Model implementation that uses fal.ai's API for inference.

    This adapter allows smolagents to work with fal.ai's language models
    by converting between smolagents' message format and fal.ai's API format.

    Parameters:
        model_id (str): The fal.ai model identifier (e.g., "fal-ai/any-llm")
        fal_model_name (str, optional): The specific LLM to use via fal.ai
            (e.g., "anthropic/claude-3.7-sonnet", "openai/gpt-4o").
            Defaults to "google/gemini-2.5-flash-lite".
        api_key (str, optional): fal.ai API key. If not provided, will use FAL_KEY
            environment variable.
        temperature (float, optional): Controls randomness (0-2). Defaults to 0.7.
        max_tokens (int, optional): Maximum tokens to generate. Defaults to 2048.
        **kwargs: Additional keyword arguments passed to fal.ai API.

    Example:
        ```python
        from smolagents import CodeAgent
        from src.models import FalAIModel

        model = FalAIModel(
            model_id="fal-ai/any-llm",
            fal_model_name="anthropic/claude-3.7-sonnet",
            temperature=0.7,
            max_tokens=4096
        )

        agent = CodeAgent(tools=[], model=model)
        result = agent.run("What is 2 + 2?")
        ```
    """

    def __init__(
        self,
        fal_api: str = "fal-ai/any-llm",
        fal_model_name: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(fal_api=fal_api, **kwargs)

        # Store fal.ai specific parameters
        self.fal_api = fal_api
        self.fal_model_name = fal_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configure fal_client if api_key provided
        if api_key:
            fal_client.api_key = api_key

        logger.info(f"Initialized FalAIModel with model: {fal_model_name}")

    def _convert_messages_to_fal_format(
        self,
        messages: list[ChatMessage | dict]
    ) -> tuple[str, str | None]:
        """
        Convert smolagents message list to fal.ai format.

        Args:
            messages: List of ChatMessage objects or dicts from smolagents

        Returns:
            tuple: (prompt, system_prompt) for fal.ai API
        """
        # Clean and standardize messages
        clean_messages = get_clean_message_list(messages)

        system_prompt = None
        user_messages = []

        for msg in clean_messages:
            role = msg["role"]
            content = msg["content"]

            # Extract text content
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content_text = "\n".join(text_parts)
            else:
                content_text = str(content)

            # Separate system prompts from user/assistant messages
            if role == "system":
                if system_prompt is None:
                    system_prompt = content_text
                else:
                    system_prompt += "\n" + content_text
            elif role == "user":
                user_messages.append(f"User: {content_text}")
            elif role == "assistant":
                user_messages.append(f"Assistant: {content_text}")

        # Combine all messages into a single prompt
        prompt = "\n\n".join(user_messages)

        return prompt, system_prompt

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Generate a response using fal.ai API.

        Args:
            messages: List of message dictionaries or ChatMessage objects
            stop_sequences: Not supported by fal.ai (ignored)
            response_format: Not supported by fal.ai (ignored)
            tools_to_call_from: Not supported by fal.ai (ignored)
            **kwargs: Additional parameters for fal.ai API

        Returns:
            ChatMessage: The model's response wrapped in smolagents format
        """
        # Convert messages to fal.ai format
        prompt, system_prompt = self._convert_messages_to_fal_format(messages)

        # Prepare fal.ai request
        fal_input = {
            "prompt": prompt,
            "model": self.fal_model_name,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if system_prompt:
            fal_input["system_prompt"] = system_prompt

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens"] and value is not None:
                fal_input[key] = value

        logger.debug(f"Calling fal.ai with input: {fal_input}")

        try:
            # Call fal.ai API using subscribe
            result = fal_client.subscribe(
                self.fal_api,
                arguments=fal_input
            )

            # Extract response
            output_text = result.get("output", "")

            # Note: fal.ai doesn't provide token usage info in response
            # We'll estimate or leave it as None
            token_usage = None

            logger.debug(f"Received response from fal.ai: {output_text[:100]}...")

            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=output_text,
                raw=result,
                token_usage=token_usage,
            )

        except Exception as e:
            logger.error(f"Error calling fal.ai API: {e}")
            raise

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta, None, None]:
        """
        Generate a streaming response using fal.ai API.

        Args:
            messages: List of message dictionaries or ChatMessage objects
            stop_sequences: Not supported by fal.ai (ignored)
            response_format: Not supported by fal.ai (ignored)
            tools_to_call_from: Not supported by fal.ai (ignored)
            **kwargs: Additional parameters for fal.ai API

        Yields:
            ChatMessageStreamDelta: Streaming response chunks
        """
        # Convert messages to fal.ai format
        prompt, system_prompt = self._convert_messages_to_fal_format(messages)

        # Prepare fal.ai request
        fal_input = {
            "prompt": prompt,
            "model": self.fal_model_name,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if system_prompt:
            fal_input["system_prompt"] = system_prompt

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens"] and value is not None:
                fal_input[key] = value

        logger.debug(f"Streaming from fal.ai with input: {fal_input}")

        try:
            # Call fal.ai API using stream
            stream = fal_client.stream(
                self.model_id,
                arguments={"input": fal_input}
            )

            # Track tokens (approximate since fal.ai doesn't provide this)
            is_first_chunk = True
            output_tokens = 0

            for event in stream:
                output_tokens += 1

                # Extract content from event
                if isinstance(event, dict):
                    content = event.get("output", "")
                    partial = event.get("partial", False)
                else:
                    # If event is not a dict, convert to string
                    content = str(event)
                    partial = True

                # Only include input tokens in first chunk
                token_usage = TokenUsage(
                    input_tokens=0 if not is_first_chunk else 0,  # fal.ai doesn't provide this
                    output_tokens=1 if not is_first_chunk else output_tokens,
                )
                is_first_chunk = False

                yield ChatMessageStreamDelta(
                    content=content,
                    tool_calls=None,
                    token_usage=token_usage,
                )

        except Exception as e:
            logger.error(f"Error streaming from fal.ai API: {e}")
            raise

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the model to a dictionary for serialization.

        Returns:
            dict: Dictionary representation of the model
        """
        model_dict = super().to_dict()
        model_dict.update({
            "fal_model_name": self.fal_model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return model_dict

    @classmethod
    def from_dict(cls, model_dict: dict[str, Any]) -> "FalAIModel":
        """
        Create a FalAIModel from a dictionary.

        Args:
            model_dict: Dictionary containing model parameters

        Returns:
            FalAIModel: New instance created from the dictionary
        """
        return cls(**model_dict)