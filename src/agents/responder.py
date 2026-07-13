"""Responder Agent - generates grounded, cited answers from retrieved context.

Responsibilities:
- Format retrieved chunks into a clear context for the LLM
- Generate an answer with match scores and citations
- Handle retry feedback from the Validator
"""

import json

from src.agents.context import PipelineContext
from src.agents.llm_client import call_llm_sync

RESPONDER_SYSTEM_PROMPT = """You are the Responder agent in an Agentic RAG system for CV expertise matching.

Your job is to generate a clear, grounded answer based ONLY on the retrieved CV chunks.

## Rules:

1. **Cite your sources.** Every claim about a person MUST reference the source document.
2. **Use retrieved data only.** Do NOT invent skills, experience, or facts not in the context.
3. **Calculate match scores.** For each candidate, estimate how well they match the query requirements as a percentage (0-100%).
4. **Be specific.** Quote relevant experience, skills, and achievements from the CVs.
5. **Handle edge cases:**
   - If no good matches exist (all below 50%), say so clearly.
   - If partial matches exist, present them with their scores.
   - If the retrieved context is insufficient, acknowledge the limitation.
6. **Structure your answer clearly:**
   - List candidates ranked by match score (best first)
   - For each candidate: name, match percentage, key qualifications, source citation
   - If fewer than 3 candidates, note that fewer matches were found
7. **Limit to Top 3 Matches:** You MUST limit the listed candidates in BOTH your human-readable answer text and in the `matches` JSON list to at most the **top 3 best matching candidates**. Completely ignore any candidates beyond the top 3 best matches.

## Match Score Guidelines:
- 90-100%: Almost perfect match - all key requirements met
- 75-89%: Strong match - most requirements met, minor gaps
- 60-74%: Good match - core requirements met, some gaps
- 50-59%: Marginal match - significant gaps
- Below 50%: Poor match - do not include unless explicitly asked

## Output Format:

Respond with a JSON object:
{
  "answer": "Human-readable answer with at most the top 3 ranked candidates...",
  "matches": [
    {
      "person_name": "Extracted from source filename",
      "score": 85,
      "evidence": "Key qualifications and experience that justify this score",
      "source_document": "filename.pdf",
      "sections": ["work_experience", "technical_skills"]
    }
  ],
  "total_candidates_found": 2,
  "has_good_match": true
}"""

RESPONDER_USER_PROMPT = """Based on the following retrieved context from the CV knowledge base, answer this query:

Query: {query}

Required skills: {skills}

Retrieved Context:
{context}

{retry_instruction}

Generate your answer as JSON."""


def respond(ctx: PipelineContext, api_key: str | None = None) -> PipelineContext:
    """Run the Responder agent.

    Generates an answer with citations and match scores.
    Incorporates Validator feedback on retries.
    """
    if not ctx.retrieved_chunks:
        ctx.answer = "No matching candidates found in the knowledge base. The knowledge base may be empty or the query did not match any documents."
        ctx.match_candidates = []
        return ctx

    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(ctx.retrieved_chunks):
        context_parts.append(
            f"[Chunk {i+1}, Source: {chunk.source}, Section: {chunk.section}, "
            f"Similarity: {chunk.similarity:.3f}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    # Build retry instruction
    retry_instruction = ""
    if ctx.retry_count > 0 and ctx.validation_feedback:
        retry_instruction = (
            f"IMPORTANT: Your previous answer had issues. Feedback: {ctx.validation_feedback}\n"
            "Fix the specific problems mentioned above. Do NOT repeat the same mistakes."
        )

    # Build user prompt
    user_prompt = RESPONDER_USER_PROMPT.format(
        query=ctx.query,
        skills=", ".join(ctx.extracted_skills) if ctx.extracted_skills else "Not specified",
        context=context,
        retry_instruction=retry_instruction,
    )

    # Call LLM (uses settings.rag_model - Validator uses settings.validation_model)
    response = call_llm_sync(
        prompt=user_prompt,
        system_prompt=RESPONDER_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=2048,
        api_key=api_key,
    )

    if not response.success:
        raise RuntimeError(f"Responder LLM call failed: {response.error or 'Unknown error'}")

    try:
        from src.utils.json_parser import parse_json_robust
        from src.agents.context import MatchCandidate
        content = response.content.strip()
        result = parse_json_robust(content)
        ctx.answer = result.get("answer", "")
        ctx.citations = result.get("matches", [])[:3]  # Strictly limit to top 3 matches
        
        ctx.match_candidates = []
        for m in ctx.citations:
            if isinstance(m, dict):
                try:
                    score = float(m.get("score", 0))
                except (ValueError, TypeError):
                    score = 0.0
                ctx.match_candidates.append(
                    MatchCandidate(
                        person_name=m.get("person_name", "Unknown"),
                        score=score,
                        evidence=m.get("evidence", ""),
                        source_document=m.get("source_document", ""),
                        sections=m.get("sections", []),
                    )
                )

    except Exception as e:
        raise RuntimeError(f"Failed to parse Responder JSON response: {str(e)}. Content was: {response.content}")

    return ctx
