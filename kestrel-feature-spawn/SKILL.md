# SpawnFeature & BridgeFeature

> Multi-agent delegation and external gateway integration.

## Spawn Skills

### spawn_agent
- **Description**: Create a new child agent with a specific purpose and constraints
- **Category**: agent_management
- **Parameters**:
  - `name` (string, required): Unique name for the child agent
  - `purpose` (string, required): What the child agent is for
  - `budget` (float, optional): Budget allocation (default 0)
  - `ttl` (int, optional): Time-to-live in seconds (default 3600)
  - `constraints` (string, optional): Comma-separated constraints
  - `features` (string, optional): Comma-separated allowed features
- **Returns**: Dict with spawned status, child name, DID, purpose, and TTL

### list_children
- **Description**: List all active child agents spawned by this agent
- **Category**: agent_management
- **Returns**: Dict with children list and count

### delegate_task
- **Description**: Send a task to an existing child agent for processing
- **Category**: agent_management
- **Parameters**:
  - `child_name` (string, required): Name of the child agent
  - `task` (string, required): Task description for the child
- **Returns**: Dict with delegation status

### get_child_result
- **Description**: Retrieve the result from a child agent's completed task
- **Category**: agent_management
- **Parameters**:
  - `child_name` (string, required): Name of the child agent
- **Returns**: Dict with result or status

### terminate_child
- **Description**: Terminate a child agent and release resources
- **Category**: agent_management
- **Parameters**:
  - `child_name` (string, required): Name of the child agent
- **Returns**: Dict with termination status

## Bridge Skills

### bridge_status
- **Description**: Show bridge configuration and connection status
- **Category**: system
- **Returns**: Dict with uptime, session counts, and config

### bridge_connections
- **Description**: List active bridge connections/sessions
- **Category**: system
- **Parameters**:
  - `limit` (int, optional): Max sessions to return (default 20)
- **Returns**: Dict with session list

### bridge_history
- **Description**: Show recent bridge invocation history
- **Category**: system
- **Parameters**:
  - `limit` (int, optional): Max entries to return (default 20)
- **Returns**: Dict with invocation log entries

## Dependencies

- Requires: kestrel-sovereign
- Optional: none
