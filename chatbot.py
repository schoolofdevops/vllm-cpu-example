#!/usr/bin/env python3
"""
Gradio Chatbot for vLLM
A simple web-based chatbot interface that connects to vLLM using OpenAI-compatible API.

Requirements:
    pip install gradio openai

Usage:
    python chatbot.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from openai import OpenAI
import sys
import os

# Configuration from environment variables
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8009/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")

# Initialize OpenAI client pointing to vLLM
try:
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="dummy"  # vLLM doesn't require authentication
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)


def chat(message, history, temperature, max_tokens, system_prompt):
    """
    Chat function that processes user messages and returns AI responses.

    Args:
        message: Current user message
        history: List of (user_msg, bot_msg) tuples
        temperature: Sampling temperature (0.0 - 2.0)
        max_tokens: Maximum tokens to generate
        system_prompt: System message to set chatbot behavior

    Yields:
        Partial responses as they're generated (streaming)
    """
    # Build messages list for OpenAI API
    messages = []

    # Add system prompt if provided
    if system_prompt and system_prompt.strip():
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Convert history from tuples to message format
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        # Create streaming completion
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        # Stream response token by token
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                yield response

    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nMake sure vLLM is running at {VLLM_BASE_URL}"
        yield error_msg


# Create Gradio interface
with gr.Blocks(title="vLLM Chatbot") as demo:
    gr.Markdown(
        """
        # 🤖 vLLM Chatbot

        Chat with SmolLM2 running on your local vLLM server.

        **Model:** {model}
        **Endpoint:** {endpoint}
        """.format(model=MODEL_NAME, endpoint=VLLM_BASE_URL)
    )

    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500
            )

            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here and press Enter...",
                lines=2
            )

            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

        with gr.Column(scale=1):
            # Settings panel
            gr.Markdown("### Settings")

            system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="You are a helpful assistant.",
                value="You are a helpful AI assistant.",
                lines=3
            )

            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Higher = more creative, Lower = more focused"
            )

            max_tokens = gr.Slider(
                minimum=50,
                maximum=500,
                value=200,
                step=10,
                label="Max Tokens",
                info="Maximum length of response"
            )

            gr.Markdown("### Quick Actions")
            example_btn1 = gr.Button("💡 Explain Docker", size="sm")
            example_btn2 = gr.Button("💻 Write Python code", size="sm")
            example_btn3 = gr.Button("📝 Write a poem", size="sm")

    # Examples section
    gr.Markdown("### Example Questions")
    gr.Examples(
        examples=[
            ["What is vLLM and how does it work?"],
            ["Explain the difference between Docker and Kubernetes."],
            ["Write a Python function to check if a number is prime."],
            ["What are the benefits of using containers?"],
            ["How do I optimize Docker images for production?"],
        ],
        inputs=msg,
    )

    # Info section
    with gr.Accordion("ℹ️ About", open=False):
        gr.Markdown(
            """
            This chatbot demonstrates local LLM inference using:
            - **vLLM**: High-performance inference engine
            - **SmolLM2**: Small but capable language model
            - **Gradio**: Interactive web interface
            - **OpenAI API**: Compatible endpoint

            ### Tips:
            - Adjust temperature for creativity vs. consistency
            - Use system prompts to guide the chatbot's behavior
            - Lower max_tokens for faster responses
            - Higher max_tokens for detailed answers

            ### Troubleshooting:
            If you get connection errors:
            1. Check vLLM is running: `docker compose ps`
            2. Verify health: `curl http://localhost:8009/health`
            3. Check logs: `docker compose logs -f vllm-cpu`
            """
        )

    # Event handlers
    def respond(message, history, temp, tokens, sys_prompt):
        """Handle chat submission and return updated history."""
        # Build messages for API
        messages = []

        if sys_prompt and sys_prompt.strip():
            messages.append({"role": "system", "content": sys_prompt})

        # Add history (already in dict format from Gradio)
        if history:
            messages.extend(history)

        # Add current message
        messages.append({"role": "user", "content": message})

        try:
            # Call API
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stream=True
            )

            # Build new history with user message and streaming response
            new_history = history + [{"role": "user", "content": message}]
            response = ""

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
                    # Yield updated history with assistant response
                    yield new_history + [{"role": "assistant", "content": response}]

        except Exception as e:
            # Log error to console for debugging
            print(f"ERROR: {type(e).__name__}: {str(e)}")
            # Provide helpful error message to user
            error_msg = f"Connection Error: {str(e)}\n\nPlease check:\n1. vLLM is running: docker compose ps\n2. API endpoint: {VLLM_BASE_URL}"
            new_history = history + [{"role": "user", "content": message}]
            yield new_history + [{"role": "assistant", "content": error_msg}]

    # Submit on button click or Enter key
    msg.submit(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens, system_prompt],
        outputs=chatbot
    ).then(
        lambda: "",  # Clear input after submission
        outputs=msg
    )

    submit.click(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens, system_prompt],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=msg
    )

    # Clear conversation
    clear.click(lambda: None, outputs=chatbot)

    # Example button handlers
    example_btn1.click(
        lambda: "Explain what Docker is and how it works.",
        outputs=msg
    )

    example_btn2.click(
        lambda: "Write a Python function to calculate the factorial of a number with error handling.",
        outputs=msg
    )

    example_btn3.click(
        lambda: "Write a haiku about artificial intelligence and containers.",
        outputs=msg
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Starting vLLM Chatbot Interface")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {VLLM_BASE_URL}")
    print("\nChecking vLLM connection...")

    # Test connection before starting
    try:
        models = client.models.list()
        print(f"✓ Connected successfully!")
        print(f"✓ Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"✗ Warning: Could not connect to vLLM")
        print(f"  Error: {e}")
        print(f"\n  Make sure vLLM is running:")
        print(f"    docker compose ps")
        print(f"    curl {VLLM_BASE_URL.replace('/v1', '')}/health")
        print(f"\n  Continuing anyway - you can start vLLM later.")

    print("\n" + "=" * 60)
    print("Starting Gradio interface...")
    print("=" * 60)

    # Launch Gradio
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )
