#!/bin/bash

# This script bundles all project source code into a single markdown file.

OUTPUT_FILE="project_code_bundle.md"
echo "# Kestrel Project Code Bundle" > "$OUTPUT_FILE"
echo "_Generated on $(date)_" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find all relevant source files
find . -type f \( \
    -name "*.py" -or \
    -name "*.js" -or \
    -name "*.html" -or \
    -name "*.toml" -or \
    -name "Dockerfile" \
\) \
-not -path "./.venv/*" \
-not -path "./agent_data/*" \
-not -name "*_bundle.md" \
| sort | while read -r code_file; do
    # Determine the language for syntax highlighting
    lang=""
    extension="${code_file##*.}"
    case "$extension" in
        py) lang="python" ;;
        js) lang="javascript" ;;
        html) lang="html" ;;
        toml) lang="toml" ;;
        Dockerfile) lang="dockerfile" ;;
    esac

    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "## File: \`$code_file\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "\`\`\`$lang" >> "$OUTPUT_FILE"
    cat "$code_file" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "✅ Code bundle created at $OUTPUT_FILE" 