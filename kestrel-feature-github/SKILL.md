# GitHubFeature

> GitHub integration for issues, PRs, repository access, and code introspection.

## Skills

### read_github_file
- **Description**: Read a file from a GitHub repository
- **Category**: data_access
- **Parameters**:
  - `repo` (string, required): Repository in 'owner/repo' format, or 'self'
  - `path` (string, required): Path to file within the repository
  - `ref` (string, optional): Branch, tag, or commit SHA (default: main)

### list_github_files
- **Description**: List files in a GitHub repository directory
- **Category**: data_access
- **Parameters**:
  - `repo` (string, required): Repository in 'owner/repo' format, or 'self'
  - `path` (string, optional): Directory path (empty for root)
  - `ref` (string, optional): Branch, tag, or commit SHA
  - `recursive` (boolean, optional): If true, list all files recursively

### search_github_code
- **Description**: Search for code in GitHub repositories
- **Category**: data_access
- **Parameters**:
  - `query` (string, required): Search query
  - `repo` (string, optional): Limit to specific repo
  - `path` (string, optional): Limit to path prefix
  - `extension` (string, optional): Limit to file extension
  - `max_results` (integer, optional): Maximum results (default 20)

### get_code_definition
- **Description**: Get a function or class definition from a Python file using AST
- **Category**: data_access
- **Parameters**:
  - `repo` (string, required): Repository or 'self'
  - `path` (string, required): Path to Python file
  - `name` (string, required): Function or class name
  - `ref` (string, optional): Branch, tag, or commit SHA

### list_code_definitions
- **Description**: List all functions and classes in a Python file
- **Category**: data_access
- **Parameters**:
  - `repo` (string, required): Repository or 'self'
  - `path` (string, required): Path to Python file
  - `ref` (string, optional): Branch, tag, or commit SHA

### get_self_repo_info
- **Description**: Get information about the agent's own source repository
- **Category**: data_access

### list_source_components
- **Description**: List all feature components in the agent's source code
- **Category**: data_access
- **Parameters**:
  - `include_files` (boolean, optional): Include file listings for each component

### get_component_source
- **Description**: Get all source files for a specific feature component
- **Category**: data_access
- **Parameters**:
  - `component` (string, required): Component name
  - `include_content` (boolean, optional): Include file contents

### invalidate_github_cache
- **Description**: Invalidate cached GitHub content to force fresh fetch
- **Category**: data_access
- **Parameters**:
  - `repo` (string, required): Repository to invalidate (or 'self')
  - `path` (string, optional): Specific path to invalidate

### list_github_issues
- **Description**: List issues in a GitHub repository
- **Category**: data_access
- **Parameters**:
  - `repo` (string, optional): Repository or 'self' (default: self)
  - `state` (string, optional): Issue state filter (default: open)
  - `labels` (string, optional): Comma-separated label names
  - `max_results` (integer, optional): Maximum results (default 30)

### get_github_issue
- **Description**: Get details of a specific GitHub issue
- **Category**: data_access
- **Parameters**:
  - `issue_number` (integer, required): Issue number
  - `repo` (string, optional): Repository or 'self' (default: self)

### get_github_issue_comments
- **Description**: Get comments on a specific GitHub issue
- **Category**: data_access
- **Parameters**:
  - `issue_number` (integer, required): Issue number
  - `repo` (string, optional): Repository or 'self' (default: self)
  - `max_results` (integer, optional): Maximum comments (default 30)

## Dependencies

- Requires: kestrel-sovereign, httpx, pyyaml, aiosqlite
