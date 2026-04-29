---
name: troubleshooting
description: Diagnose product issues using the experience knowledge base and product specs. Use when the user reports a problem, error, or malfunction.
license: MIT
compatibility: requires RAGFlow server with experience KB populated
metadata:
  agent: deep_rag
  version: "1.0"
allowed-tools: ragflow_list_datasets ragflow_retrieve get_next_chunks get_kb_datasets_by_type
---

# Troubleshooting Skill

## When to Use

- User reports a product malfunction or error
- User describes unexpected behavior
- User asks "why is X happening" or "how do I fix Y"
- User asks about error codes or warning messages
- User needs to diagnose an issue based on symptoms

## Workflow

### Step 1: Gather symptoms

Extract key information from the user's description:
- Product model number
- What exactly is happening
- When the issue started
- Any error codes or warning lights
- Environmental factors (temperature, usage time, etc.)

### Step 2: Match model

Use `complete_model_number` to resolve the full model number if needed.

### Step 3: Search experience KB

Use `get_kb_datasets_by_type` with the experience KB type, then `ragflow_retrieve` to find similar past cases in the experience KB (`经验库`).

### Step 4: Cross-reference with product KB

Search the product KB (`产品库`) for the model's specifications and known limitations.

### Step 5: Diagnose

Based on retrieved content:
1. Identify the most likely cause(s)
2. Check past case resolution rates
3. Provide step-by-step solution(s)
4. Note any precautions or when to seek professional help

### Step 6: Suggest follow-up

If the issue is unresolved:
- Ask for more specific symptoms
- Suggest checking specific components
- Recommend professional service if needed

## Answer Format

```
**问题诊断**：{root cause analysis}

**解决方案**：
1. {step 1} — {success rate from past cases if known}
2. {step 2}
...

**注意事项**：{precautions or when to escalate}
```

## Important Rules

- Always search experience KB first — past cases are most valuable
- Be honest when no similar cases exist — suggest alternative approaches
- Distinguish between user-fixable issues and those requiring professional service
- Cite past case frequency where available (e.g., "this solution worked in 80% of similar cases")
