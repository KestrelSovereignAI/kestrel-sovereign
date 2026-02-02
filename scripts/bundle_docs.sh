#!/bin/bash

# This script bundles all project documentation into a single markdown file.

OUTPUT_FILE="project_documentation_bundle.md"
echo "# Kestrel Project Documentation Bundle" > "$OUTPUT_FILE"
echo "_Generated on $(date)_" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Use find to locate all markdown files in the docs directory, respecting the hierarchy
find docs -name "*.md" | sort | while read -r doc_file; do
    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "## File: \`$doc_file\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    cat "$doc_file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "✅ Documentation bundle created at $OUTPUT_FILE" 