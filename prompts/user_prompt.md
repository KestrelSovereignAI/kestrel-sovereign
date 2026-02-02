# User Prompt Template

This template is used to format the user's query with context.

## Template Variables

- `{context}` - Situational context including relevant documents and conversation history
- `{query}` - The user's actual query

## Template

```
--- SITUATIONAL CONTEXT ---
{context}
--- END CONTEXT ---

Based on the instructions, documents, and context above, answer the following query.

Query: {query}
```
