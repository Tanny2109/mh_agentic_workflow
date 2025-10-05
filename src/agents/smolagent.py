"""Smolagents implementation for fal.ai workflow"""
import os
import re
from typing import List, Dict, Any, Generator

import gradio as gr
from smolagents import CodeAgent, InferenceClientModel

from ..tools.fal_tools import (
    FalImageGenerationTool,
    FalVideoGenerationTool,
    FalImageEditTool
)


class SmolagentFalApp:
    """Smolagents-based application for fal.ai workflow with streaming support"""
    
    def __init__(self, hf_token: str, model_id: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """Initialize the smolagent application
        
        Args:
            hf_token: Hugging Face API token
            model_id: LLM model identifier
        """
        # Initialize fal.ai tools
        self.tools = [
            FalImageGenerationTool(),
            FalVideoGenerationTool(),
            FalImageEditTool()
        ]

        # Initialize agent with Hugging Face model
        self.model = InferenceClientModel(model_id=model_id, token=hf_token)

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=10,
        )

    def parse_image_paths(self, text: str) -> List[str]:
        """Extract image paths from agent output
        
        Args:
            text: Agent output text
            
        Returns:
            List of image file paths found in text
        """
        pattern = r'(/[^\s]+\.png)'
        matches = re.findall(pattern, text)
        return matches

    def stream_agent_response(
        self, user_message: str, history: List[Dict[str, Any]]
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream agent thinking and results to Gradio chatbot
        
        Args:
            user_message: User's input message
            history: Chat history
            
        Yields:
            Updated chat history with agent responses
        """
        try:
            # Add user message to history
            history = history or []
            history.append({"role": "user", "content": user_message})
            yield history

            # Stream thinking process
            thinking_msg = {
                "role": "assistant",
                "content": "🤔 Analyzing your request..."
            }
            history.append(thinking_msg)
            yield history

            # Clear previous logs
            self.agent.logs = []

            # Run agent - it will populate agent.logs
            agent_output = self.agent.run(user_message)
            print(f"Agent output: {agent_output}")

            # Stream thinking from logs
            thinking_content = "🤔 **Agent Thinking:**\n\n"
            for log in self.agent.logs:
                if isinstance(log, dict):
                    # Display task/thought
                    if 'task' in log:
                        thinking_content += f"**Task:** {log['task']}\n\n"
                    # Display LLM output (thinking)
                    if 'llm_output' in log:
                        thinking_content += f"**Thought:** {log['llm_output']}\n\n"
                    # Display code being executed
                    if 'tool_calls' in log:
                        for tool_call in log['tool_calls']:
                            thinking_content += f"**Tool:** `{tool_call.get('name', 'unknown')}`\n"
                    # Display rationale
                    if 'rationale' in log:
                        thinking_content += f"{log['rationale']}\n\n"

                history[-1]["content"] = thinking_content
                yield history

            # Parse for images
            image_paths = self.parse_image_paths(str(agent_output))

            if image_paths:
                # Build content with text and images
                content = []

                # Add text if agent output contains more than just the image path
                if len(str(agent_output).strip()) > len(' '.join(image_paths)):
                    content.append({
                        "type": "text",
                        "text": f"**Agent Response:**\n\n{agent_output}"
                    })

                # Add all images
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        content.append({"path": img_path, "mime_type": "image/png"})

                # Use list only if we have multiple items
                if len(content) == 1:
                    history[-1]["content"] = content[0]
                else:
                    history[-1]["content"] = content
            else:
                # No images, just text
                history[-1]["content"] = f"**Agent Response:**\n\n{agent_output}"

            yield history

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your request."
            history.append({"role": "assistant", "content": error_msg})
            yield history

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface
        
        Returns:
            Gradio Blocks interface
        """
        custom_css = """
        .chatbot-container {
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        """

        with gr.Blocks(css=custom_css, title="Fal.ai Agent") as demo:
            gr.Markdown("""
            # 🤖 Fal.ai Agentic Workflow with Smolagents

            This agent can:
            - 🎨 Generate images using fal.ai models (nano-banana, flux-schnell, flux-pro)
            - 🎬 Create videos with text descriptions
            - ✏️ Edit and transform uploaded images
            - 💭 Stream thinking process and results

            **Examples:**
            - "Generate 2 images of a sunset over mountains using flux-pro"
            - "Create a 5-second video of a cat playing with yarn"
            - "Make the image more colorful and vibrant"
            """)

            chatbot = gr.Chatbot(
                type="messages",
                label="Agent Chat",
                height=600
            )

            with gr.Row():
                chat_input = gr.MultimodalTextbox(
                    placeholder="Ask me to generate images or videos...",
                    show_label=False,
                    file_count="single"
                )

            # Examples
            gr.Examples(
                examples=[
                    "Generate an image of a futuristic city with flying cars",
                    "Create 2 images of a dragon, make them vertical format",
                    "Generate a video of ocean waves at sunset, 5 seconds long"
                ],
                inputs=chat_input
            )

            # Advanced settings
            with gr.Accordion("📊 Agent Info", open=False):
                gr.Markdown("""
                **Available Tools:**
                - `fal_image_generation`: Generate images using fal.ai models
                - `fal_video_generation`: Create videos from text
                - `fal_image_edit`: Edit and transform images

                **How it works:**
                1. Agent analyzes your request
                2. Selects appropriate fal.ai tool and parameters
                3. Executes the tool
                4. Returns results with images/videos inline

                **Supported Models:**
                - nano-banana (fast, high quality)
                - flux-schnell (fast)
                - flux-pro (highest quality)
                """)

            def handle_input(message: Dict[str, Any], history: List[Dict[str, Any]]):
                """Handle both text and file inputs"""
                text_input = message.get("text", "")
                files = message.get("files", [])

                # If file uploaded, add to message
                if files:
                    file_path = files[0] if isinstance(files, list) else files
                    text_input = f"{text_input}\n\nUploaded file: {file_path}"

                # Stream response
                yield from self.stream_agent_response(text_input, history)

            chat_input.submit(
                handle_input,
                inputs=[chat_input, chatbot],
                outputs=[chatbot]
            )

        return demo
