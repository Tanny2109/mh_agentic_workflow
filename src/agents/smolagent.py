"""Smolagents implementation for fal.ai workflow"""
import os
import re
from typing import List, Dict, Any, Generator
import fal_client

import gradio as gr
from smolagents import CodeAgent
from ..models import FalAIModel
from dotenv import load_dotenv
load_dotenv()

from ..tools.fal_tools import (
    FalImageGenerationTool,
    FalVideoGenerationTool,
    FalImageEditTool
)


class SmolagentFalApp:
    """Smolagents-based application for fal.ai workflow with streaming support"""
    
    def __init__(self, hf_token: str, fal_model_name="google/gemini-2.5-flash"):
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
        self.model = FalAIModel(fal_model_name=fal_model_name)

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
        pattern = r'(/[^\s]+\.(?:png|jpg|jpeg))'
        matches = re.findall(pattern, text)
        return matches

    def parse_video_paths(self, text: str) -> List[str]:
        """Extract video paths from agent output

        Args:
            text: Agent output text

        Returns:
            List of video file paths found in text
        """
        pattern = r'(/[^\s]+\.(?:mp4|mov|avi|webm))'
        matches = re.findall(pattern, text)
        return matches

    def _build_context_from_history(self, history: List[Dict[str, Any]]) -> str:
        """Build conversation context from chat history

        Args:
            history: Chat history

        Returns:
            Formatted conversation context string
        """
        # Get last few exchanges for context (last 4 messages = 2 exchanges)
        recent_history = history[-5:-1] if len(history) > 1 else []

        context_parts = []
        if recent_history:
            context_parts.append("Previous conversation:")
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Extract text from content if it's a dict/list
                if isinstance(content, dict):
                    content = content.get("text", str(content))
                elif isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                    content = " ".join(text_parts) if text_parts else str(content)
                context_parts.append(f"{role.capitalize()}: {content}")

        # Add current user message
        current_msg = history[-1].get("content", "") if history else ""
        if isinstance(current_msg, str):
            context_parts.append(f"\nCurrent request: {current_msg}")
        else:
            context_parts.append(f"\nCurrent request: {current_msg}")

        return "\n".join(context_parts)

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

            # Clear previous logs
            self.agent.logs = []

            # Build conversation context from history (BEFORE adding thinking message)
            conversation_context = self._build_context_from_history(history)

            # Stream thinking process (AFTER building context)
            thinking_msg = {
                "role": "assistant",
                "content": "🤔 Analyzing your request..."
            }
            history.append(thinking_msg)
            yield history

            # Track steps for streaming updates
            last_step_count = 0
            step_start_times = {}
            total_start_time = None

            # Run agent step by step with streaming
            print(f"\n{'='*80}")
            print(f"DEBUG - Running agent with context: {conversation_context[:200]}...")
            print(f"{'='*80}\n")

            try:
                import threading
                import time

                agent_output = None
                agent_error = None

                def run_agent():
                    nonlocal agent_output, agent_error
                    try:
                        agent_output = self.agent.run(conversation_context)
                    except Exception as e:
                        agent_error = e

                # Start agent in background thread
                agent_thread = threading.Thread(target=run_agent)
                agent_thread.start()
                total_start_time = time.time()

                # Spinner frames for animation
                spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                spinner_idx = 0

                # Stream updates while agent is running
                while agent_thread.is_alive():
                    # Check if there are new logs or steps
                    current_step_count = len(self.agent.memory.steps)
                    elapsed_time = time.time() - total_start_time

                    # Update spinner
                    spinner = spinner_frames[spinner_idx % len(spinner_frames)]
                    spinner_idx += 1

                    # Build thinking content from current state
                    thinking_content = f"{spinner} **Processing...**\n\n"

                    # Show elapsed time
                    thinking_content += f"⏱️ *Elapsed: {elapsed_time:.1f}s*\n\n"

                    # Show completed steps with timing
                    for idx, step in enumerate(self.agent.memory.steps):
                        step_time = step_start_times.get(idx, 0)
                        thinking_content += f"**Step {idx + 1}** *(took {step_time:.1f}s)*\n"

                        # Show what tool was used
                        if hasattr(step, 'tool_calls') and step.tool_calls:
                            for tool_call in step.tool_calls:
                                if hasattr(tool_call, 'arguments'):
                                    args = str(tool_call.arguments)
                                    # Detect what's happening
                                    if 'fal_image_generation' in args:
                                        thinking_content += f"   🎨 Generating image...\n"
                                    elif 'fal_image_edit' in args:
                                        thinking_content += f"   ✏️ Editing image...\n"
                                    elif 'fal_video_generation' in args:
                                        thinking_content += f"   🎬 Generating video...\n"
                                    elif 'final_answer' in args:
                                        thinking_content += f"   ✨ Finalizing...\n"

                        # Show result preview
                        if hasattr(step, 'action_output'):
                            output_preview = str(step.action_output)[:80]
                            if len(str(step.action_output)) > 80:
                                output_preview += "..."
                            thinking_content += f"   `{output_preview}`\n"

                        thinking_content += "\n"

                    # Track new step start time
                    if current_step_count > last_step_count:
                        step_start_times[last_step_count] = time.time() - total_start_time
                        if last_step_count > 0:
                            # Calculate time for previous step
                            prev_step_time = step_start_times[last_step_count] - step_start_times.get(last_step_count - 1, 0)
                            step_start_times[last_step_count - 1] = prev_step_time

                    # Show current activity with spinner
                    if current_step_count < self.agent.max_steps:
                        thinking_content += f"\n{spinner} **Step {current_step_count + 1}** in progress...\n"

                        # Estimate time remaining (rough estimate)
                        if current_step_count > 0:
                            avg_step_time = elapsed_time / current_step_count
                            estimated_remaining = avg_step_time * (self.agent.max_steps - current_step_count)
                            thinking_content += f"   *Estimated remaining: ~{estimated_remaining:.0f}s*\n"

                    history[-1]["content"] = thinking_content
                    yield history
                    last_step_count = current_step_count

                    time.sleep(0.3)  # Poll every 300ms for updates

                # Wait for thread to complete
                agent_thread.join()

                # Calculate final step time
                if last_step_count > 0 and last_step_count not in step_start_times:
                    step_start_times[last_step_count] = time.time() - total_start_time - step_start_times.get(last_step_count - 1, 0)

                if agent_error:
                    raise agent_error

                print(f"\n{'='*80}")
                print(f"DEBUG - Agent run completed successfully")
                print(f"DEBUG - Agent output: {agent_output}")
                print(f"DEBUG - Agent output type: {type(agent_output)}")
                print(f"DEBUG - Agent memory steps: {len(self.agent.memory.steps)}")
                print(f"{'='*80}\n")

            except Exception as e:
                print(f"\n{'='*80}")
                print(f"ERROR - Agent run failed: {e}")
                print(f"{'='*80}\n")
                raise

            # Extract tool outputs from memory steps
            tool_outputs = []
            for idx, step in enumerate(self.agent.memory.steps):
                print(f"DEBUG - Examining step {idx}: {type(step).__name__}")

                # Check action_output
                if hasattr(step, 'action_output'):
                    action_output = step.action_output
                    print(f"DEBUG - Step {idx} action_output: {action_output}")
                    if action_output and "Generated" in str(action_output) and ".png" in str(action_output):
                        tool_outputs.append(str(action_output))
                        print(f"DEBUG - Added action_output to tool_outputs: {action_output}")

                # Check observations
                if hasattr(step, 'observations'):
                    observations = step.observations
                    print(f"DEBUG - Step {idx} observations: {observations}")
                    if observations and "Generated" in str(observations) and ".png" in str(observations):
                        tool_outputs.append(str(observations))
                        print(f"DEBUG - Added observations to tool_outputs: {observations}")

                # Check tool_calls for debugging
                if hasattr(step, 'tool_calls'):
                    print(f"DEBUG - Step {idx} tool_calls: {step.tool_calls}")

            # Use tool outputs if available, otherwise use agent_output
            # Combine all tool outputs to get complete result
            combined_output = "\n".join(tool_outputs) if tool_outputs else str(agent_output)
            print(f"DEBUG - Combined output for parsing: {combined_output}")

            # Parse for media files (images and videos) from the combined tool outputs
            '''
            So when we generate more than one image, this gets tricky. agent output is somehow dependent on how i add history. if i make multiple
            additions to history by looping through images, I dont get any images, but when i restrict it to one image and adding this to
            ChatMessage, this gets displayed. Maybe some thing to do with gradio. Ill look into this later on.
            '''
            image_paths = self.parse_image_paths(combined_output)
            video_paths = self.parse_video_paths(combined_output)

            if image_paths:
                # Display image
                history[-1]["content"] = {
                    "path": image_paths[0]
                }
            elif video_paths:
                # Display video
                history[-1]["content"] = {
                    "path": video_paths[0]
                }
            else:
                # No media files, just text
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
                    file_count="single",
                    interactive=True
                )

            submit_btn = gr.Button("Send", variant="primary")

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
                for updated_history in self.stream_agent_response(text_input, history):
                    yield updated_history, None  # Clear input after submission

            # Connect both submit and button click
            submit_events = [
                chat_input.submit(
                    handle_input,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                ),
                submit_btn.click(
                    handle_input,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )
            ]

            # Add loading state to button and input
            for event in submit_events:
                event.then(
                    lambda: gr.Button(value="Send", interactive=True),
                    None,
                    submit_btn
                )

        return demo
