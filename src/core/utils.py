"""Utility functions for streaming agent responses to Gradio"""
from __future__ import annotations

import os
import re
from typing import Generator

from gradio import ChatMessage
from smolagents import CodeAgent


def parse_image_paths(text: str) -> list[str]:
    """Extract valid image paths from agent output
    
    Args:
        text: Agent output text potentially containing image paths
        
    Returns:
        List of valid image file paths
    """
    pattern = r'(/[^\s]+\.png)'
    matches = re.findall(pattern, text)
    return [path for path in matches if os.path.exists(path)]


def pull_message_from_step(step_log: dict) -> Generator[ChatMessage, None, None]:
    """Extract ChatMessages from a single agent step log
    
    Args:
        step_log: Dictionary containing agent step information
        
    Yields:
        ChatMessage objects for each component of the step
    """
    # Display task
    if step_log.get("task"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "📋 Task"},
            content=step_log["task"]
        )

    # Display rationale/thinking
    if step_log.get("rationale"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "💭 Thinking"},
            content=step_log["rationale"]
        )

    # Display LLM output
    if step_log.get("llm_output"):
        yield ChatMessage(
            role="assistant",
            content=step_log["llm_output"]
        )

    # Display tool calls
    if step_log.get("tool_calls"):
        for tool_call in step_log["tool_calls"]:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("arguments", {})

            content = f"```json\n{tool_args}\n```"
            yield ChatMessage(
                role="assistant",
                metadata={"title": f"🛠️ Using tool: {tool_name}"},
                content=content
            )

    # Display observations/results
    if step_log.get("observation"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "👁️ Observation"},
            content=f"```\n{step_log['observation']}\n```"
        )

    # Display errors
    if step_log.get("error"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "💥 Error"},
            content=str(step_log["error"])
        )


def stream_from_smolagent(
    agent: CodeAgent, prompt: str
) -> Generator[ChatMessage, None, ChatMessage | None]:
    """Run a smolagent and stream messages as ChatMessages
    
    Args:
        agent: The CodeAgent instance to run
        prompt: User's input prompt
        
    Yields:
        ChatMessage objects for each step of the agent's execution
        
    Returns:
        Final ChatMessage with the agent's output (text and/or images)
    """
    # Clear previous logs
    agent.logs = []

    # Initial thinking message
    yield ChatMessage(
        role="assistant",
        content="🤔 Analyzing your request..."
    )

    # Run agent - it populates agent.logs
    agent_output = agent.run(prompt)
    print(f"Agent output: {agent_output}")

    # Stream thinking from logs
    for step_log in agent.logs:
        if isinstance(step_log, dict):
            for message in pull_message_from_step(step_log):
                yield message

    # Parse output for images
    image_paths = parse_image_paths(str(agent_output))

    # Build final response
    if image_paths:
        # Check if output is just image paths or has additional text
        output_text = str(agent_output).strip()
        has_extra_text = len(output_text) > len(' '.join(image_paths))

        if len(image_paths) == 1 and not has_extra_text:
            # Single image, no extra text
            yield ChatMessage(
                role="assistant",
                content={"path": image_paths[0], "mime_type": "image/png"}
            )
        else:
            # Multiple images or images with text
            if has_extra_text:
                yield ChatMessage(
                    role="assistant",
                    metadata={"title": "✅ Final Answer"},
                    content=output_text
                )

            # Yield each image
            for img_path in image_paths:
                yield ChatMessage(
                    role="assistant",
                    content={"path": img_path, "mime_type": "image/png"}
                )
    else:
        # No images, just text
        yield ChatMessage(
            role="assistant",
            metadata={"title": "✅ Final Answer"},
            content=str(agent_output)
        )

    return None
