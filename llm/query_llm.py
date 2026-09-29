"""
llm/query_llm.py

Local, offline LLM layer for Phase 6+ evidence -- rephrasing and grounded
Q&A ONLY. This model NEVER sees raw transactions and NEVER makes a risk
judgment; it only rephrases or answers questions using the evidence, score
breakdown, and graph facts the deterministic pipeline already computed.

STRICT GROUNDING -- four independent layers, because a system prompt alone
is a soft constraint small models can drift under:

  1. Pre-filter (deterministic, no LLM call): if the question doesn't
     reference the entity or any forensic/evidence vocabulary at all, refuse
     immediately. Zero hallucination risk for obviously off-topic asks --
     and saves a model call.
  2. System prompt with CONCRETE few-shot refusal examples, not just an
     abstract "don't hallucinate" instruction -- small models follow
     demonstrated patterns far better than abstract rules.
  3. Stateless single-turn calls: each call gets ONLY the system prompt +
     that one evidence dict + that one question. No conversation history,
     no way for anything outside the passed context to leak in.
  4. Post-hoc numeric verification (deterministic, reuses the exact
     approach from evaluation/metrics.py's Phase 8 evidence-accuracy
     check): every number the model's response contains is extracted and
     cross-checked against numbers that actually appear in the evidence
     text. Any number that doesn't trace back to the evidence fails the
     response outright -- it is not shown to the user, a safe fallback is
     shown instead. This is the real safety net; layers 1-3 reduce how
     often it has to fire.

Model file expected at: llm/models/<filename>.gguf (see llm/README.md).
Requires: pip install llama-cpp-python
"""

from __future__ import annotations

import glob
import os
import re

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FORENSIC_KEYWORDS = {
    "risk", "score", "evidence", "wallet", "entity", "transaction", "cluster",
    "graph", "neighbor", "neighbour", "fee", "anomaly", "flagged", "community",
    "centrality", "connected", "behavior", "behaviour", "suspicious", "chain",
    "amount", "fan", "critical", "high", "medium", "low", "why", "reason",
}

SYSTEM_PROMPT = (
    "You are a forensic evidence assistant. You will be given structured evidence "
    "already computed by a deterministic pipeline (risk score, evidence statements, "
    "graph facts). Your ONLY job is to rephrase this evidence clearly or answer "
    "questions about it.\n\n"
    "STRICT RULES:\n"
    "1. NEVER state a fact, number, or statistic that is not present in the provided evidence.\n"
    "2. NEVER assert that an entity is guilty, criminal, or definitely engaged in "
    "wrongdoing -- risk scores are prioritization signals, not conclusions.\n"
    "3. If a question is not answerable from the provided evidence, say so explicitly. "
    "Do not guess, infer beyond the evidence, or answer general-knowledge questions.\n"
    "4. Keep responses concise and factual.\n\n"
    "EXAMPLES OF CORRECT BEHAVIOR:\n"
    "Q: \"What's the capital of France?\"\n"
    "A: \"That's outside the evidence I have access to -- I can only answer questions "
    "about this entity's forensic evidence.\"\n\n"
    "Q: \"Is this wallet definitely involved in money laundering?\"\n"
    "A: \"I can't make that determination. The evidence shows [cite specific evidence "
    "items], which raised this entity's risk score, but that's a prioritization signal "
    "for investigation, not a conclusion of wrongdoing.\"\n\n"
    "Q: \"What's this entity's exact home address?\"\n"
    "A: \"That information isn't part of the evidence provided to me.\""
)


def find_model_file() -> str | None:
    matches = sorted(glob.glob(os.path.join(MODELS_DIR, "*.gguf")))
    return matches[0] if matches else None


def is_plausibly_answerable(question: str, entity_explanation: dict) -> bool:
    """Layer 1: deterministic pre-filter. Refuses obviously off-topic
    questions BEFORE spending a model call on them. This is intentionally
    permissive (low false-refusal risk) -- it only blocks questions with
    zero overlap with the entity or forensic vocabulary at all; borderline
    cases still reach the LLM, which is instructed (layer 2) and checked
    (layer 4) to refuse appropriately itself."""
    q_lower = question.lower()
    entity_id = entity_explanation.get("entity_id", "")

    if entity_id and entity_id.lower()[:12] in q_lower:
        return True
    if any(kw in q_lower for kw in FORENSIC_KEYWORDS):
        return True
    return False


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def extract_numbers(text: str) -> list[float]:
    """Pull every numeric token out of a string, e.g. '87.3%' -> [87.3]."""
    return [float(m) for m in _NUMBER_RE.findall(text)]


def verify_response_grounded(response_text: str, entity_explanation: dict, tolerance: float = 0.15) -> dict:
    """Layer 4: the real safety net. Every number in the model's response
    must trace back to a number actually present in the evidence text
    (within a small tolerance for rounding). A response introducing ANY
    untraceable number is rejected outright."""
    evidence_text = " ".join(e["statement"] for e in entity_explanation.get("evidence", []))
    evidence_text += f" {entity_explanation.get('risk_score', '')} {entity_explanation.get('risk_level', '')}"
    evidence_numbers = extract_numbers(evidence_text)

    response_numbers = extract_numbers(response_text)
    untraceable = [
        n for n in response_numbers
        if not any(abs(n - e) <= tolerance for e in evidence_numbers)
    ]

    return {
        "grounded": len(untraceable) == 0,
        "response_numbers": response_numbers,
        "evidence_numbers": evidence_numbers,
        "untraceable_numbers": untraceable,
    }


SAFE_FALLBACK = "I don't have verifiable evidence to answer that -- showing the raw evidence instead is safer than a possibly-inaccurate summary."


def strip_think_tags(raw: str) -> str:
    """Qwen3 (and other 'thinking' models) emit a <think>...</think>
    reasoning trace before the actual answer -- the investigator wants the
    answer, not the model's scratchpad. Handled as regex post-processing,
    not a prompt instruction, since prompts asking a model not to think are
    unreliable and thinking mode can be a model-level default.

    Two cases:
      1. Properly closed <think>...</think> -- remove the tag AND its
         enclosed reasoning entirely.
      2. An unclosed <think> (no matching </think> -- e.g. generation cut
         off mid-thought before max_tokens caught up): there's no reliable
         way to know where reasoning ends and any real answer begins, so
         only the literal tag markers are stripped, leaving whatever text
         is there visible rather than silently discarding it."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    return cleaned if cleaned else raw.replace("<think>", "").replace("</think>", "").strip()


class EvidenceLLM:
    def __init__(self, model_path: str | None = None, n_ctx: int = 4096, n_threads: int | None = None):
        from llama_cpp import Llama

        model_path = model_path or find_model_file()
        if model_path is None:
            raise FileNotFoundError(
                f"No .gguf model file found in {MODELS_DIR}. See llm/README.md."
            )
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def _chat(self, user_content: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = result["choices"][0]["message"]["content"].strip()
        return strip_think_tags(raw)

    def summarize_case_file(self, entity_explanation: dict) -> str:
        evidence_lines = "\n".join(f"- [{e['category']}] {e['statement']}" for e in entity_explanation["evidence"])
        prompt = (
            f"Entity: {entity_explanation['entity_id']}\n"
            f"Risk score: {entity_explanation['risk_score']} ({entity_explanation['risk_level']})\n"
            f"Evidence:\n{evidence_lines}\n\n"
            f"Write a 3-5 sentence case-file summary of this entity using ONLY the evidence above."
        )
        response = self._chat(prompt)
        check = verify_response_grounded(response, entity_explanation)
        return response if check["grounded"] else SAFE_FALLBACK

    def answer_question(self, question: str, entity_explanation: dict) -> str:
        if not is_plausibly_answerable(question, entity_explanation):
            return (
                "That question doesn't appear related to this entity's forensic evidence, "
                "so I won't guess at an answer. Ask about the risk score, evidence, or "
                "graph/behavioral findings for this entity instead."
            )

        evidence_lines = "\n".join(f"- [{e['category']}] {e['statement']}" for e in entity_explanation["evidence"])
        prompt = (
            f"Entity: {entity_explanation['entity_id']}\n"
            f"Risk score: {entity_explanation['risk_score']} ({entity_explanation['risk_level']})\n"
            f"Available evidence:\n{evidence_lines}\n\n"
            f"Question: {question}\n"
            f"Answer using ONLY the evidence above. If it doesn't answer the question, say so."
        )
        response = self._chat(prompt)
        check = verify_response_grounded(response, entity_explanation)
        return response if check["grounded"] else SAFE_FALLBACK


if __name__ == "__main__":
    model_path = find_model_file()
    if model_path is None:
        print(f"No .gguf file found in {MODELS_DIR}")
        print("See llm/README.md for setup instructions.")
    else:
        print(f"Found model: {model_path}")
        print("Run a real query with EvidenceLLM(...).summarize_case_file(...) once loaded.")

