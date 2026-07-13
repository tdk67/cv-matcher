"""Validator Agent - checks answer quality and provides feedback for retry.

Responsibilities:
- Verify answer is grounded in retrieved context
- Check that citations are present and correct
- Validate match scores align with evidence
- Provide specific, actionable feedback for retry
"""

import json

from src.agents.context import PipelineContext
from src.agents.llm_client import call_llm_sync
from src.config import settings

VALIDATOR_SYSTEM_PROMPT = """You are the Validator agent in an Agentic RAG system for CV expertise matching.

Your job is to check if the generated answer meets quality standards.

## Validation Criteria:

1. **GROUNDED**: Every claim must be supported by the retrieved context chunks
   - Flag: claims about skills, experience, companies, or education not in the context
   
2. **CITED**: Each candidate recommendation must reference the source document
   - Flag: candidates listed without source filename
   
3. **CORRECT**: Match percentages must be reasonable given the evidence
   - Flag: high scores (90%+) with weak evidence, or low scores with strong evidence
   
4. **COMPLETE**: The answer must address all parts of the query
   - Flag: query asks for 3 skills but answer only addresses 1
   
5. **SCOPE**: The answer must stay within the CV/expertise domain
   - Flag: answer includes opinions, recommendations, or personal judgments

## Failure Types:

- "hallucinated": Answer contains claims not in retrieved context
- "missing_citation": Candidates listed without source references
- "wrong_match": Match scores don't align with evidence
- "incomplete": Answer doesn't fully address the query
- "out_of_scope": Answer goes beyond CV matching

## Output Format:

Respond with a JSON object:
{
  "passed": true | false,
  "failure_type": "hallucinated" | "missing_citation" | "wrong_match" | "incomplete" | "out_of_scope" | "none",
  "specific_issues": ["issue 1", "issue 2"],
  "suggested_fix": "Concrete instruction for fixing the issues, or empty if passed"
}"""

VALIDATOR_USER_PROMPT = """Validate this answer against the retrieved context and query.

Query: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Check for hallucinations, missing citations, incorrect scores, incompleteness, and scope violations.
Return your validation as JSON."""


def validate(ctx: PipelineContext, api_key: str | None = None) -> PipelineContext:
    """Run the Validator agent.

    Checks the Responder's answer for quality issues.
    Sets validation_passed and provides feedback for retry.
    """
    if not ctx.answer:
        ctx.validation_passed = False
        ctx.validation_feedback = "No answer generated."
        ctx.validation_failure_type = "incomplete"
        return ctx

    # If no retrieved chunks, skip validation (fallback answer)
    if not ctx.retrieved_chunks:
        ctx.validation_passed = True
        ctx.validation_feedback = ""
        return ctx

    # Build context string
    context_parts = []
    for i, chunk in enumerate(ctx.retrieved_chunks):
        context_parts.append(
            f"[Chunk {i+1}, Source: {chunk.source}, Section: {chunk.section}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    # Call LLM for validation
    user_prompt = VALIDATOR_USER_PROMPT.format(
        query=ctx.query,
        context=context,
        answer=ctx.answer,
    )

    response = call_llm_sync(
        prompt=user_prompt,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=1024,
        model=settings.validation_model,
        api_key=api_key,
    )

    if not response.success:
        raise RuntimeError(f"Validator LLM call failed: {response.error or 'Unknown error'}")

    try:
        from src.utils.json_parser import parse_json_robust
        content = response.content.strip()
        result = parse_json_robust(content)
        ctx.validation_passed = result.get("passed", True)
        ctx.validation_feedback = result.get("suggested_fix", "")
        ctx.validation_failure_type = result.get("failure_type", "none")

    except Exception as e:
        raise RuntimeError(f"Failed to parse Validator JSON response: {str(e)}. Content was: {response.content}")

    return ctx
