#!/usr/bin/env python3
"""
Base class for Kestrel Application Extensions.
"""
from typing import Optional, Dict, Any

class AppExtension:
    """
    Base class for creating application-specific extensions for the Kestrel Agent.
    Extensions can modify prompts, handle custom commands, or change agent behavior
    based on the application context.
    """

    def __init__(self, agent):
        self._agent = agent

    def pre_process_input(self, user_input: str) -> Optional[str]:
        """
        Hook that runs before the main input processing.
        If it returns a string, that string is returned as the response immediately,
        bypassing the normal LLM call.
        """
        return None

    def post_process_response(self, response: str, metadata: Dict[str, Any]) -> str:
        """
        Hook that runs after the LLM has generated a response.
        It can modify the final response before it's sent to the user.
        """
        return response

    def get_system_prompt_prefix(self) -> str:
        """
        Returns a prefix to be added to the system prompt for this app context.
        """
        return ""

    def get_constitution_amendments(self) -> str:
        """
        Returns a string of amendments to be appended to the core Kestrel Constitution.
        """
        return "" 