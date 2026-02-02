# PRD: Multi-Model Foundational Support

## 1. Vision

To evolve Kestrel from a single-model agent into a resilient, flexible, and powerful multi-model agent. This will allow Kestrel to leverage the best available foundation models based on user preference, cost, and availability, ensuring it is not locked into any single provider. Ollama will serve as the default, private, "last resort" option, guaranteeing the agent can always reason.

## 2. Architecture

This will be achieved through an abstraction layer for LLM providers and a configuration-driven, prioritized fallback system. Fallbacks respect privacy, enabling human-led apps without external risks.

-   **FR1: LLM Adapter Abstraction:**
    -   A new abstract base class, `LLMAdapter`, is implemented in the `llm` module.
    -   It defines a standard interface: `generate_response(prompt: str) -> str` and `is_available() -> bool`.
    -   Concrete classes like `OllamaAdapter`, `OpenAIAdapter`, etc., implement this interface.

-   **FR2: Configuration-Driven Setup:**
    -   A new configuration file, `llm_config.toml`, will allow the user to define their providers, API keys, and model choices.
    -   **This file will be added to `.gitignore` to protect sensitive API keys.**
    -   The user will define a `provider_priority` list in the config (e.g., `["openai", "anthropic", "ollama"]`) to control the fallback order.

-   **FR3: Resilient Fallback Logic:**
    -   The `LLMService` manages an ordered list of `LLMAdapter` instances based on the `provider_priority`.
    -   When reasoning is required, the service iterates through the adapters in order.
    -   If an adapter fails, the failure is logged, and the service automatically tries the next adapter in the list.

## 3. Architectural Diagram

```mermaid
graph TD
    subgraph "Your Configuration (llm_config.toml)"
        A["priority = ['openai', 'anthropic', 'ollama']"]
        B["openai<br/>api_key='...'<br/>model='gpt-5'"]
        C["anthropic<br/>api_key='...'<br/>model='claude-3-opus'"]
        D["ollama<br/>host='...'"]
    end

    subgraph "Kestrel's Reasoning Process"
        E[User Query] --> F{Fallback Handler};
        F -- "1. Try" --> G(OpenAI Engine);
        F -- "2. On Fail, Try" --> H(Anthropic Engine);
        F -- "3. On Fail, Try" --> I(Ollama Engine);
        G -- "Success!" --> J[Agent Response];
        H -- "Success!" --> J;
        I -- "Success!" --> J;
    end
    
    style G fill:#9f9
    style H fill:#f9f
    style I fill:#cde
```

## 4. Implementation Status ✅

1.  ✅ Created the `llm` directory.
2.  ✅ Implemented `llm/adapter.py` with the `LLMAdapter` abstract base class.
3.  ✅ Implemented `llm/ollama_adapter.py` for local Ollama models.
4.  ✅ Implemented `llm/openai_adapter.py` for OpenAI API access.
5.  ✅ Created sample `llm_config.toml.example` to guide users.
6.  ✅ Updated `.gitignore` to ignore the user's `llm_config.toml`.
7.  ✅ Implemented `llm/service.py` with fallback logic and provider management.
8.  ✅ Integrated LLMService into `KestrelAgent` and `main.py` with configuration loading. 