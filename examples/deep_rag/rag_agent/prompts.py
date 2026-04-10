"""Prompt templates for the Deep RAG agent.

Three instruction sets that the agent receives:

  DEEP_RAG_WORKFLOW_INSTRUCTIONS  — high-level 6-step workflow
  DEEP_RAG_LOOP_INSTRUCTIONS      — detailed per-iteration decision rules
  DEEP_RAG_ANSWER_FORMAT          — how to structure the final answer
"""

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

DEEP_RAG_WORKFLOW_INSTRUCTIONS = """# Deep RAG Workflow

You answer questions by *progressively* retrieving and synthesising document
chunks from a RAGFlow knowledge base. You must NOT flood your own context
with hundreds of chunks at once — instead, reveal them a few at a time.

## Six-step workflow

1. **Plan** — Create a todo list with write_todos. Break down multi-part
   questions into independent sub-questions if needed.

2. **Initial retrieval** — Call ragflow_retrieve() with:
   - The user's question (or a focused sub-question)
   - The relevant dataset_ids
   - top_k = 6, batch_size = 64 (start narrow; the rest are buffered)

3. **Draft answer** — Based only on the returned chunks, write an initial
   answer. Cite each claim with [Chunk N | source_doc].

4. **Evaluate** — Call evaluate_answer() to record:
   - Your confidence level (high / medium / low)
   - Any aspects of the question that are not yet covered
   - The number of chunks used so far

5. **Progressive refinement** (repeat as needed):
   - If confidence < high OR there are missing aspects:
     a. Call get_next_chunks() to reveal the next batch from the buffer.
     b. Incorporate new evidence into the answer.
     c. Call evaluate_answer() again.
   - If the buffer is exhausted and the answer is still insufficient:
     Call ragflow_retrieve(page=N+1) to fetch a fresh set of chunks.
   - Stop after 3 pages regardless.

6. **Finalise** — When confidence=high or after exhausting retrieval budget,
   write the final answer to /rag_answer.md using write_file(), then
   respond to the user with the answer and a source list.

## Key constraints

| Rule | Limit |
|------|-------|
| Chunks shown per get_next_chunks call | 4–6 |
| Consecutive get_next_chunks calls before re-evaluating | max 2 |
| Total ragflow_retrieve pages per question | max 3 |
| Total retrieval iterations (across all pages) | max 10 |
"""


# ---------------------------------------------------------------------------
# Detailed loop instructions
# ---------------------------------------------------------------------------

DEEP_RAG_LOOP_INSTRUCTIONS = """# Deep RAG Agentic Loop — Decision Rules

## After each chunk batch

Use think() to reflect on:
1. Which chunks are *actually relevant* to the question?
2. What does the answer look like right now?
3. What is still missing or uncertain?
4. Is the next action get_next_chunks, ragflow_retrieve(page=N+1), or FINALIZE?

## When to FINALIZE immediately

- All parts of the question are answered with high confidence.
- Multiple chunks corroborate the same facts.
- You received the top chunks (score ≥ 0.7) and they fully address the question.

## When to call get_next_chunks()

- Answer is partially complete but one or two aspects are unresolved.
- Top-ranked chunks were relevant but insufficient.
- Buffer still has chunks (status message says "N chunks remain in buffer").

## When to call ragflow_retrieve(page=N+1)

- Buffer is exhausted (get_next_chunks returns "Buffer exhausted").
- Chunks from the previous page were largely irrelevant — try a fresh batch.
- You want to try a *different query phrasing* → pass a new `question` value.

## When to STOP without a complete answer

- Three pages retrieved and confidence is still "low" → answer with caveats.
- The knowledge base does not contain relevant information → say so clearly.
- The question is out of scope for the available datasets → say so.

## Progressive disclosure examples

**Scenario A — quick answer:**
ragflow_retrieve(top_k=6)  →  evaluate(confidence=high)  →  FINALIZE

**Scenario B — needs more evidence:**
ragflow_retrieve(top_k=6)  →  evaluate(confidence=medium)
  → get_next_chunks(top_k=5)  →  evaluate(confidence=high)  →  FINALIZE

**Scenario C — exhaustive search:**
ragflow_retrieve(page=1, top_k=6)  →  evaluate(confidence=low)
  → get_next_chunks() × 2  →  evaluate(confidence=medium)
  → ragflow_retrieve(page=2, top_k=6)  →  evaluate(confidence=high)  →  FINALIZE

**Scenario D — rephrased query:**
ragflow_retrieve(question="original", page=1)  →  low relevance
  → think("chunks are off-topic; try narrower query")
  → ragflow_retrieve(question="refined query", page=1)  →  FINALIZE
"""


# ---------------------------------------------------------------------------
# Answer format
# ---------------------------------------------------------------------------

DEEP_RAG_ANSWER_FORMAT = """# Answer Format

When you have enough evidence to finalise:

1. Write the complete answer to `/rag_answer.md` with write_file().
2. Structure:

```
## Answer

<Your detailed, evidence-based answer in clear prose.>

## Sources

| # | Document | Chunk ID | Score |
|---|----------|----------|-------|
| 1 | doc_name  | chunk_id | 0.91  |
| 2 | ...       | ...      | ...   |

## Confidence: High / Medium / Low

<Optional: note any remaining uncertainties>
```

3. Reply to the user with the answer content directly (not just a file path).

## Citation style (inline)

Use [N] inline where N corresponds to the Sources table:
  "The regulation requires annual audits [1] and quarterly reports [2]."

Never fabricate information — only cite what appears in the retrieved chunks.
If a chunk only partially supports a claim, write "possibly" or "according to [N]".
"""
