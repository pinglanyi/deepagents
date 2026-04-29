---
name: knowledge-base
description: Manage RAGFlow knowledge bases — create, list, update, delete datasets, upload and parse documents. Use when the user asks to manage the knowledge base, upload documents, or configure datasets.
license: MIT
compatibility: requires RAGFlow server with API key
metadata:
  agent: deep_rag
  version: "1.0"
allowed-tools: ragflow_list_datasets ragflow_retrieve
---

# Knowledge Base Management Skill

## When to Use

- User wants to create a new knowledge base / dataset
- User wants to upload documents to a dataset
- User asks about dataset status or document listing
- User wants to delete or update a dataset
- User needs to parse (process) uploaded documents

## Available Operations

### List Datasets
Use `ragflow_list_datasets` to see all available knowledge bases. Filter by name with the `name_filter` parameter.

### Upload Documents
Upload files to a dataset. Supported formats: PDF, DOCX, TXT, MD, images, etc.

### Parse Documents
After uploading, documents must be parsed to become searchable. Parse can be triggered per-dataset.

### Manage Documents
List, update, or delete documents within a dataset.

## KB Types in This System

The Deep RAG system recognizes these KB types by dataset name:

| KB Type | Dataset Name Matches | Purpose |
|---|---|---|
| Product KB | `产品库` | Product Q&A, specs, features |
| Image KB | `图片库` | Images with descriptions |
| File KB | `文件库` | Documents, manuals, PDFs |
| Video KB | `视频库` | Video links and descriptions |
| General KB | `通用库` | Industry standards, general knowledge |
| Program KB | `程序库` | Program/function parameters |
| Experience KB | `经验库` | Troubleshooting cases, history |

## Important Rules

- Before uploading, confirm the dataset exists with `ragflow_list_datasets`
- Do NOT upload files without user confirmation
- Always trigger parsing after upload
- Large batches: upload in groups and wait for parsing between batches
- Parsing failures: check file format is supported
