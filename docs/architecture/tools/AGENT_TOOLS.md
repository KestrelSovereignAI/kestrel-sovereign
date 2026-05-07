# Agent Tools Documentation

> **⚠ DEPRECATED — describes a removed architecture.**
> This doc covers the `AgentToolMixin` / `kestrel_agent_tools.py` /
> top-level `tools/` directory pattern, all of which have been
> removed. Tool delivery now happens through feature packages
> registered via the `kestrel_sovereign.features` entry-point group
> using the `@tool` decorator from `kestrel_sdk.features.base`.
>
> See [`docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md`](../core/FEATURE_AGENT_FRAMEWORK.md)
> for the modern pattern.
>
> Rewrite tracked in [#1047](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1047);
> kept here meanwhile for git-archaeology context.

## Overview

Kestrel agents now have access to powerful tools that enhance their capabilities:

1. **Web Search** - Real-time web search using Tavily API
2. **Feedback & Diagnostics** - Self-observation and error tracking
3. **Tool Usage Analytics** - Automatic tracking of tool performance

These tools enable agents to:
- Search the web for current information
- Record their own observations and diagnostics
- Track errors and capability gaps
- Provide transparency into their tool usage

## Configuration

### Environment Variables

```bash
# Required for web search
export TAVILY_API_KEY="your-tavily-api-key"

# Database (required for feedback tools)
export DATABASE_URL="postgresql://user:pass@host:port/dbname"
```

### Getting a Tavily API Key

1. Sign up at https://tavily.com
2. Get your API key from the dashboard
3. Add it to your `.env` file or export it

## Available Tools

### 1. Web Search Tool

Enables agents to perform real-time web searches and access current information.

**Commands:**
```
!search <query>         - Search the web
!web-search <query>     - Alternative search command
```

**Examples:**
```
!search latest developments in AI
!web-search weather in San Francisco
```

**Features:**
- AI-generated answer summaries
- Top 5 relevant results with URLs
- Content snippets from each result
- Automatic search tracking in database

**Response Format:**
```
🔍 Web Search Results for: 'your query'

Summary: [AI-generated summary of findings]

Found 5 results:

1. [Title]
   URL: [URL]
   [Content snippet...]

2. [Title]
   ...

(Search completed in XXXms)
```

### 2. Feedback & Diagnostic Tool

Allows agents to record observations, track errors, and maintain self-diagnostics.

**Commands:**
```
!feedback list [type] [severity]   - List feedback entries
!feedback stats                     - View tool usage statistics
!feedback record <type> <title> <content> - Manually record feedback
```

**Feedback Types:**
- `observation` - General observations about user behavior or context
- `diagnostic` - Self-diagnostic about internal state or performance
- `tool_error` - Errors encountered while using tools
- `user_feedback` - Feedback from the user
- `self_reflection` - Agent's reflections on interactions
- `capability_gap` - Identifies missing capabilities

**Severity Levels:**
- `info` - Informational only
- `warning` - Potential issue
- `error` - Error occurred
- `critical` - Critical error

### 3. Model Management Tools

Tools for managing the agent's underlying AI models (via Ollama).

**Commands:**
```
!list-models            - List all available models
!pull-model <name>      - Download a new model (e.g., !pull-model llama3)
!model-info <name>      - Get detailed info about a model
!storage-status         - Check model storage usage
!cleanup-models         - Remove unused models to free space
```

### 4. MCP (Model Context Protocol) Tools

Tools for dynamically loading and using external capabilities via Docker containers.

**Commands:**
```
!mcp-load <image>       - Load an MCP server from a Docker image
!mcp-list               - List active MCP servers and tools
!mcp-call <container> <tool> [json_args] - Execute a specific MCP tool
!mcp-unload <container> - Stop an MCP server
```

### 5. Sovereignty Tools

Tools for managing the agent's sovereign identity and data export.

**Commands:**
```
!export-sovereignty [tier] - Export agent state to IPFS/Filecoin
!import-sovereignty <cid>  - Restore agent state from an export
!sovereignty-status        - Check export status and history
```


**Examples:**
```
!feedback list                                    # List all feedback
!feedback list tool_error error                   # List tool errors
!feedback stats                                   # View usage statistics
!feedback record observation "User preference" "User prefers concise answers"
```

**Response Formats:**

List feedback:
```
📝 Found 3 feedback entries:

⏳ [ERROR] Web search failed
   Type: tool_error | Source: tool
   Created: 2025-11-07 15:30:00
   Search failed: Connection timeout...

✅ [INFO] User prefers technical details
   Type: observation | Source: agent
   Created: 2025-11-07 14:15:00
   User consistently asks for code examples...
```

Tool stats:
```
📊 Tool Usage Statistics (Last 7 days):

🔧 web_search
   Total uses: 15 | Success rate: 93.3%
   Successful: 14 | Failed: 1
   Avg time: 850ms | Max time: 1500ms

🔧 feedback_tool
   Total uses: 8 | Success rate: 100.0%
   Successful: 8 | Failed: 0
   Avg time: 25ms | Max time: 50ms
```

### 3. Tool Discovery

View all available tools and their status.

**Command:**
```
!tools    - List all available tools
```

**Example Response:**
```
🔧 Available Tools:

🔍 Web Search: ✅ Enabled
   Commands: !search <query>, !web-search <query>
   Powered by: Tavily API

📝 Feedback & Diagnostics: ✅ Enabled
   Commands: !feedback list|stats|record
   Features: Self-diagnostics, observations, tool error tracking
```

### 4. MCP Tool Manager (Coming Soon)

Enables agents to dynamically discover, install, and use tools from the Docker MCP Hub.

**Commands:**
```
!mcp-search <query>     - Search for tools on Docker Hub
!mcp-install <image>    - Install an MCP server image
!mcp-list               - List installed MCP servers
!mcp-start <image>      - Start an MCP server
!mcp-stop <image>       - Stop an MCP server
```

**Features:**
- Access to the entire ecosystem of Model Context Protocol tools
- Sandboxed execution in Docker containers
- Dynamic capability expansion without code changes

## Programmatic Tool Access

### For Agent Developers

Agents can use tools programmatically (not just via commands):

```python
# Web search
result = await agent.web_search.search_and_format("query", max_results=5)

# Record observation
await agent.record_observation(
    title="User interaction pattern",
    content="User frequently asks follow-up questions",
    severity=FeedbackSeverity.INFO,
    tags=["interaction", "pattern"]
)

# Record capability gap
await agent.record_capability_gap(
    missing_capability="image generation",
    context="User asked for diagram creation",
    workaround="Provided text description instead"
)

# Track tool usage
await agent.record_tool_usage(
    tool_name="custom_tool",
    success=True,
    execution_time_ms=250,
    input_params={"param": "value"},
    output_result={"result": "data"}
)
```

## Database Schema

### agent_feedback Table
Stores all feedback, observations, and diagnostics.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| companion_id | UUID | Reference to companion |
| user_id | UUID | Reference to user |
| feedback_type | VARCHAR(50) | Type of feedback |
| severity | VARCHAR(20) | Severity level |
| source | VARCHAR(50) | Source (agent/user/system/tool) |
| title | VARCHAR(255) | Short title |
| content | TEXT | Detailed content |
| context | JSONB | Additional JSON context |
| tags | TEXT[] | Searchable tags |
| is_resolved | BOOLEAN | Resolution status |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Update timestamp |

### tool_usage Table
Tracks all tool executions and performance.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| companion_id | UUID | Reference to companion |
| user_id | UUID | Reference to user |
| tool_name | VARCHAR(100) | Name of tool |
| success | BOOLEAN | Whether tool succeeded |
| input_params | JSONB | Input parameters |
| output_result | JSONB | Output/result |
| error_message | TEXT | Error if failed |
| execution_time_ms | INTEGER | Execution time |
| conversation_context | TEXT | What triggered the tool |
| created_at | TIMESTAMPTZ | Creation timestamp |

### web_searches Table
History of web searches performed by agents.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| companion_id | UUID | Reference to companion |
| user_id | UUID | Reference to user |
| query | TEXT | Search query |
| search_provider | VARCHAR(50) | Provider (tavily) |
| results | JSONB | Array of search results |
| result_count | INTEGER | Number of results |
| conversation_context | TEXT | Why search was performed |
| created_at | TIMESTAMPTZ | Creation timestamp |

## Privacy & Security

### Data Retention

- Feedback entries persist until manually resolved or deleted
- Tool usage data retained for analytics (configurable)
- Web search history retained for context (configurable)
- All data respects companion privacy modes (ephemeral, isolated, normal)

### Multi-Tenant Isolation

- All tool data is scoped to companion_id and user_id
- Row-level security ensures users only access their own data
- Agents cannot access other users' tool data or feedback

### API Key Security

- Tavily API key stored as environment variable
- Never exposed to users or logs
- Validated at tool initialization

## Best Practices

### For Agent Developers

1. **Always check tool availability** before using:
   ```python
   if hasattr(agent, 'web_search') and agent.web_search.enabled:
       result = await agent.web_search.search(query)
   ```

2. **Record capability gaps** when users ask for unavailable features:
   ```python
   await agent.record_capability_gap(
       missing_capability="image generation",
       context=user_query
   )
   ```

3. **Track tool usage** for debugging:
   ```python
   start = time.time()
   try:
       result = await some_tool()
       await agent.record_tool_usage(
           tool_name="some_tool",
           success=True,
           execution_time_ms=int((time.time() - start) * 1000)
       )
   except Exception as e:
       await agent.record_tool_usage(
           tool_name="some_tool",
           success=False,
           error_message=str(e)
       )
   ```

### For Users

1. **Use `!tools`** to see what's available
2. **Use `!feedback stats`** to understand agent tool usage
3. **Review feedback** to see what the agent has observed
4. **Provide feedback** using `!feedback record`

## Troubleshooting

### Web Search Not Working

**Issue:** `❌ Web search is not available`

**Solution:**
1. Check TAVILY_API_KEY is set: `echo $TAVILY_API_KEY`
2. Verify API key is valid at https://tavily.com
3. Restart the server after setting the key

### Feedback Tool Not Working

**Issue:** `❌ Feedback tool is not available`

**Solution:**
1. Check DATABASE_URL is set
2. Verify database connection
3. Run migrations: Database tables are auto-created on startup
4. Check logs for database errors

### Tool Commands Not Responding

**Issue:** Commands like `!search` return no response

**Solution:**
1. Verify tools are initialized: Check agent has `init_tools()` called
2. Check for errors in server logs
3. Ensure agent class is `KestrelAgent` with `AgentToolMixin`

## Future Enhancements

Planned tool additions:
- [ ] Image generation tool (DALL-E, Stable Diffusion)
- [ ] Document analysis tool (PDF, images)
- [ ] Calculator/computation tool
- [ ] Memory search tool (semantic search)
- [ ] Code execution sandbox
- [ ] File upload/download tools

## API Reference

See:
- `/tools/web_search.py` - WebSearchTool implementation
- `/tools/feedback_tool.py` - FeedbackTool implementation
- `/kestrel_agent_tools.py` - AgentToolMixin integration

---

*Last Updated: November 7, 2025*
