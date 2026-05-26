from .adapter import LLMAdapter
from kestrel_sdk.llm import ProviderCapabilities, ToolStreamingMode
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MockAdapter(LLMAdapter):
    """
    Mock adapter for demo purposes when no real LLM providers are available.
    Returns canned responses for testing the agent interface.
    """

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            tool_streaming_mode=ToolStreamingMode.NONE,
            notes=("Mock adapter returns canned text and does not exercise LLM features.",),
        )

    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        Return a mock response for demo purposes.
        """
        # Extract the user's message from the messages
        # Look for the actual user query, not the system prompts
        user_message = ""
        
        # First, try to find the conversation history in the system prompt
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
                break
        
        # Extract user query from conversation history in system prompt
        if "Conversation History:" in system_content:
            history_part = system_content.split("Conversation History:")[1].split("--- END CONTEXT ---")[0]
            # Find the last user message in history
            lines = history_part.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('user: '):
                    user_message = line[6:].strip()  # Remove 'user: ' prefix
                    break
        
        # If we didn't find it in history, fall back to the last user message content
        if not user_message:
            for msg in reversed(messages):  # Check from the end
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        # Skip if it looks like a system prompt
                        if "You are Kestrel" in content or "KESTREL CONSTITUTION" in content:
                            continue
                        user_message = content
                        break
                    elif isinstance(content, list):
                        for part in content:
                            if part.get("type") == "text":
                                text = part["text"]
                                if "You are Kestrel" not in text and "KESTREL CONSTITUTION" not in text:
                                    user_message = text
                                    break
                        if user_message:
                            break

        # Generate a simple mock response based on the input
        if "hello" in user_message.lower() or "hi" in user_message.lower():
            return "Hello! I'm a Kestrel AI agent running in a demo container. I'm designed to be your sovereign AI companion with full control over your data and interactions. How can I help you today?"
        elif "introduce" in user_message.lower() or "who are you" in user_message.lower():
            return "I'm a Kestrel sovereign AI agent. I was created to give you full ownership and control over your AI interactions. Unlike other AI systems, I store everything locally and respect your privacy. I can help with creative tasks, web searches, model management, and much more!"
        elif "what can you do" in user_message.lower():
            return "As a Kestrel agent, I can:\n\n• Generate creative content and images\n• Search the web for information\n• Manage AI models and their usage\n• Maintain your digital sovereignty\n• Learn from our conversations\n• Help with various tasks through specialized tools\n\nI'm designed to be your personal AI companion that you fully control."
        else:
            return f"I understand you said: '{user_message}'. As a demo Kestrel agent, I'm running in a container without real LLM access, but I can still demonstrate the interface! In a full deployment, I'd provide intelligent responses to help you with any task. Would you like to know more about Kestrel's features?"

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ):
        """
        Mock streaming response - just yields the full response.
        """
        response = await self.get_response(client, model, messages, **kwargs)
        yield response
