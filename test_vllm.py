#!/usr/bin/env python3
"""
Test script for vLLM CPU deployment.
Demonstrates various API usage patterns.

Requirements:
    pip install openai requests
"""

import json
import sys
import time
from typing import Optional

try:
    import requests
    from openai import OpenAI
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install openai requests")
    sys.exit(1)


class VLLMTester:
    """Test harness for vLLM API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8009"):
        self.base_url = base_url
        self.v1_url = f"{base_url}/v1"
        self.client = OpenAI(base_url=self.v1_url, api_key="dummy")

    def check_health(self) -> bool:
        """Check if vLLM service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Health check passed")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Health check failed: {e}")
            return False

    def list_models(self) -> Optional[list]:
        """List available models."""
        try:
            response = requests.get(f"{self.v1_url}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                print(f"✓ Found {len(models)} model(s):")
                for model in models:
                    print(f"  - {model.get('id')}")
                return models
            else:
                print(f"✗ Failed to list models: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to list models: {e}")
            return None

    def test_completion(self, model: str) -> bool:
        """Test basic completion endpoint."""
        print(f"\n--- Testing Completion API ---")
        try:
            prompt = "The capital of France is"
            print(f"Prompt: {prompt}")

            start_time = time.time()
            response = requests.post(
                f"{self.v1_url}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["text"]
                print(f"Response: {text}")
                print(f"✓ Completion succeeded ({elapsed:.2f}s)")
                return True
            else:
                print(f"✗ Completion failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Completion failed: {e}")
            return False

    def test_chat_completion(self, model: str) -> bool:
        """Test chat completion endpoint using OpenAI client."""
        print(f"\n--- Testing Chat Completion API ---")
        try:
            messages = [
                {"role": "user", "content": "Explain Docker in one sentence."}
            ]
            print(f"Messages: {json.dumps(messages, indent=2)}")

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0.7,
            )
            elapsed = time.time() - start_time

            content = response.choices[0].message.content
            print(f"Response: {content}")
            print(f"✓ Chat completion succeeded ({elapsed:.2f}s)")
            return True
        except Exception as e:
            print(f"✗ Chat completion failed: {e}")
            return False

    def test_streaming(self, model: str) -> bool:
        """Test streaming response."""
        print(f"\n--- Testing Streaming API ---")
        try:
            prompt = "Write a haiku about containers:"
            print(f"Prompt: {prompt}")
            print("Response: ", end="", flush=True)

            start_time = time.time()
            response = requests.post(
                f"{self.v1_url}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.8,
                    "stream": True,
                },
                stream=True,
                timeout=30,
            )

            full_text = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            text = data["choices"][0]["text"]
                            print(text, end="", flush=True)
                            full_text += text
                        except json.JSONDecodeError:
                            continue

            elapsed = time.time() - start_time
            print(f"\n✓ Streaming succeeded ({elapsed:.2f}s)")
            return True
        except Exception as e:
            print(f"\n✗ Streaming failed: {e}")
            return False

    def test_performance(self, model: str, num_requests: int = 5) -> None:
        """Test performance with multiple requests."""
        print(f"\n--- Testing Performance ({num_requests} requests) ---")
        prompts = [
            "What is machine learning?",
            "Explain neural networks.",
            "What is Docker?",
            "Define Kubernetes.",
            "What is continuous integration?",
        ]

        times = []
        for i, prompt in enumerate(prompts[:num_requests], 1):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.v1_url}/completions",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": 30,
                        "temperature": 0.7,
                    },
                    timeout=30,
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    times.append(elapsed)
                    print(f"  Request {i}: {elapsed:.2f}s")
                else:
                    print(f"  Request {i}: Failed ({response.status_code})")
            except Exception as e:
                print(f"  Request {i}: Failed ({e})")

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\nPerformance Summary:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")


def main():
    """Run all tests."""
    print("=" * 60)
    print("vLLM CPU Deployment - API Test Suite")
    print("=" * 60)

    # Initialize tester
    tester = VLLMTester()

    # Check health
    print("\n1. Health Check")
    if not tester.check_health():
        print("\n⚠️  Service is not healthy. Please check:")
        print("  - Is the container running? (docker compose ps)")
        print("  - Check logs: docker compose logs vllm-cpu")
        print("  - Wait for model to load (first startup takes 5-10 minutes)")
        sys.exit(1)

    # List models
    print("\n2. List Models")
    models = tester.list_models()
    if not models or len(models) == 0:
        print("\n⚠️  No models found. Service may still be initializing.")
        sys.exit(1)

    model_name = models[0]["id"]
    print(f"\nUsing model: {model_name}")

    # Run tests
    print("\n" + "=" * 60)
    print("Running Tests")
    print("=" * 60)

    success_count = 0
    total_tests = 0

    # Test 1: Basic completion
    total_tests += 1
    if tester.test_completion(model_name):
        success_count += 1

    # Test 2: Chat completion
    total_tests += 1
    if tester.test_chat_completion(model_name):
        success_count += 1

    # Test 3: Streaming
    total_tests += 1
    if tester.test_streaming(model_name):
        success_count += 1

    # Test 4: Performance
    tester.test_performance(model_name, num_requests=3)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {success_count}/{total_tests}")
    print(f"Failed: {total_tests - success_count}/{total_tests}")

    if success_count == total_tests:
        print("\n✓ All tests passed! vLLM is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
