# Claude Code Agent Container
# Runs autonomous GitHub issue processing using your Claude Max subscription

FROM node:22-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    jq \
    ripgrep \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN mkdir -p /usr/share/keychains \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /usr/share/keychains/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keychains/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keychains/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user for security
RUN useradd -m -s /bin/bash agent && \
    mkdir -p /home/agent/.claude && \
    mkdir -p /home/agent/.venv && \
    chown -R agent:agent /home/agent

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER agent
WORKDIR /home/agent

# Install uv for agent user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/agent/.local/bin:$PATH"

# Create container-local venv (not in workspace to avoid conflicts with host)
ENV UV_PROJECT_ENVIRONMENT=/home/agent/.venv

# Create workspace directory
RUN mkdir -p /home/agent/workspace

# Set environment for non-interactive use
ENV CLAUDE_CODE_SKIP_THEME=1
ENV CI=true

WORKDIR /home/agent/workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
