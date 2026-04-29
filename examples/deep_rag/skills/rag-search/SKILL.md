---
name: rag-search
description: Multi-source RAG retrieval across product, image, file, video, general, program, and experience knowledge bases. Use when the user asks product questions, looks for images/files/videos, or needs troubleshooting.
license: MIT
compatibility: requires RAGFlow server
metadata:
  agent: deep_rag
  version: "1.0"
allowed-tools: ragflow_list_datasets ragflow_retrieve get_next_chunks complete_model_number get_kb_datasets_by_type
---

# RAG Search Skill

## When to Use

- User asks about a product (specifications, features, pricing, comparisons)
- User looks for images, files, videos, or documents
- User needs technical parameters or program/function details
- User asks for troubleshooting or past case history
- User asks about industry standards or general knowledge

## Workflow

### Step 1: Identify the KB type

Determine which knowledge base(s) are relevant to the question:

| Question Type | KB Type | Dataset Name Contains |
|---|---|---|
| Product specs, models, features | Product KB | `产品库` |
| Image search, visual content | Image KB | `图片库` |
| Documents, PDFs, manuals | File KB | `文件库` |
| Video tutorials, demos | Video KB | `视频库` |
| Industry standards, compliance | General KB | `通用库` |
| Programming parameters, API | Program KB | `程序库` |
| Past cases, troubleshooting | Experience KB | `经验库` |

### Step 2: Match model number

If the question contains a product model number:
1. Extract it from the question
2. Optionally complete it with `complete_model_number`
3. Map the model to KB datasets via `get_kb_datasets_by_type`

### Step 3: Retrieve

Use `ragflow_list_datasets` to find matching datasets, then `ragflow_retrieve` to get content.

For targeted retrieval: pass the specific dataset IDs to `ragflow_retrieve(dataset_ids=[...])`.
For broad retrieval: call `ragflow_retrieve(dataset_ids=[])` to search across all datasets.

### Step 4: Fetch more if needed

If the initial results are insufficient, use `get_next_chunks` to fetch additional content chunks from the same search.

### Step 5: Synthesize and answer

Combine retrieved content into a clear, concise answer. Cite sources when possible. Follow the answer format rules:
- Product info: list specs, model, features
- Multimedia: describe content + provide URL
- Troubleshooting: state the problem, list solutions with steps, mention success rate from past cases

## Important Rules

- NEVER fabricate information not in the retrieved content
- If no relevant content is found, say so and suggest refining the search
- For model-specific questions, always try the specific dataset first before broad search
- Use `get_next_chunks()` at most once — don't loop endlessly
- Answer inline — do NOT write files unless explicitly asked
