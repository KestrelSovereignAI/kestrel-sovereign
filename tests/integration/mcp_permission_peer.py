"""Native external MCP peer that records actual tool side effects."""

from pathlib import Path

from mcp.server import MCPServer


def main():
    server = MCPServer('permission-boundary-witness')

    @server.tool()
    def record_marker(marker: str) -> str:
        with Path('/tmp/permission-witness').open('a') as stream:
            stream.write(marker + '\n')
        return 'Recorded: ' + marker

    server.run(transport='streamable-http', host='0.0.0.0', port=8000,
               stateless_http=True, json_response=True)


if __name__ == '__main__':
    main()
