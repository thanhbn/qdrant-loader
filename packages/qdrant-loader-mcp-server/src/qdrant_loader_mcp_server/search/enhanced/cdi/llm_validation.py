# ============================================================
# LEARNING: llm_validation.py - AIKH-488 (SPIKE-008: Validate detect_document_conflicts)
# This file has been annotated with TODO markers for learning.
# To restore: git checkout -- packages/qdrant-loader-mcp-server/src/qdrant_loader_mcp_server/search/enhanced/cdi/llm_validation.py
# MCP Tool: detect_document_conflicts
# Learning Objectives:
# - [ ] L3: Understand LLM provider abstraction for conflict validation (AIKH-488/AIKH-603: TC-CONFLICT-003)
# - [ ] L3: Understand structured prompt engineering for conflict detection (AIKH-488/AIKH-603: TC-CONFLICT-003)
# - [ ] L3: Understand LLM response parsing and validation (AIKH-488/AIKH-603: TC-CONFLICT-003)
# - [ ] LN: Understand async timeout handling for LLM calls (AIKH-488/AIKH-610: TC-CONFLICT-010)
# ============================================================

from __future__ import annotations

import asyncio
from typing import Any


async def validate_conflict_with_llm(
    detector: Any, doc1: Any, doc2: Any, similarity_score: float
) -> tuple[bool, str, float]:
    """Validate potential conflict using LLM analysis.

    Use Case: Final tier (L3) validation using LLM for conflict confirmation
    Data Flow: (doc1, doc2, similarity_score) -> LLM prompt -> parse response
    Business Rule: Returns (conflict_detected, explanation, confidence)
    Git: "LLM validation integration" - Tier 3 conflict detection
    """
    # TODO [L3]: Check LLM provider availability - AIKH-488/AIKH-603 (TC-CONFLICT-003)
    # MCP Tool: detect_document_conflicts
    # Use Case: Validate that LLM provider is configured for conflict analysis
    # Business Rule: Support both core LLM provider and legacy AsyncOpenAI client
    # Test: test_llm_validation_tier
    # -----------------------------------------------------------
    # Prefer core provider when available; fallback to AsyncOpenAI client if present
    provider = getattr(getattr(detector, "engine", None), "llm_provider", None)
    openai_client = getattr(detector, "openai_client", None)
    if provider is None and openai_client is None:
        return False, "LLM validation not available", 0.0
    # -----------------------------------------------------------

    try:
        # TODO [LN]: Configure LLM timeout from settings - AIKH-488/AIKH-610 (TC-CONFLICT-010)
        # MCP Tool: detect_document_conflicts
        # Use Case: Allow per-call customization of LLM timeout for performance tuning
        # Business Rule: Default timeout 10.0 seconds, configurable via conflict_llm_timeout_s
        # Test: test_performance_under_10_seconds
        # -----------------------------------------------------------
        settings = (
            getattr(detector, "_settings", {}) if hasattr(detector, "_settings") else {}
        )
        timeout_s = settings.get("conflict_llm_timeout_s", 10.0)
        # -----------------------------------------------------------

        # TODO [L3]: Construct conflict detection prompt - AIKH-488/AIKH-603 (TC-CONFLICT-003)
        # MCP Tool: detect_document_conflicts
        # Use Case: Build structured prompt for LLM conflict analysis
        # Business Rule: Include document titles, content excerpts (1000 chars), and similarity score
        # Data Flow: (doc1, doc2, similarity_score) -> formatted prompt string
        # Test: test_llm_validation_tier
        # -----------------------------------------------------------
        prompt = (
            "Analyze two documents for conflicts in information, recommendations, or approaches.\n"
            f"Doc1: {doc1.source_title}\nContent: {doc1.content[:1000]}...\n"
            f"Doc2: {doc2.source_title}\nContent: {doc2.content[:1000]}...\n"
            f"Vector Similarity: {similarity_score:.3f}\n"
            "Respond: CONFLICT_DETECTED|CONFIDENCE|EXPLANATION (concise)."
        )
        # -----------------------------------------------------------

        # TODO [L3]: Execute LLM call with timeout - AIKH-488/AIKH-603 (TC-CONFLICT-003)
        # MCP Tool: detect_document_conflicts
        # Use Case: Call LLM for conflict validation with configurable timeout
        # Business Rule: Support both core LLM provider and legacy AsyncOpenAI client
        # Infrastructure: asyncio.wait_for wraps LLM call for timeout control
        # Test: test_llm_validation_tier
        # -----------------------------------------------------------
        if provider is not None:
            chat_client = provider.chat()
            response = await asyncio.wait_for(
                chat_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=(
                        getattr(
                            getattr(detector, "_settings", {}), "get", lambda *_: None
                        )("conflict_llm_model", None)
                        or "gpt-3.5-turbo"
                    ),
                    max_tokens=200,
                    temperature=0.1,
                ),
                timeout=timeout_s,
            )
            # Normalize text extraction
            content = (response or {}).get("text", "")
        else:
            raw = await asyncio.wait_for(
                openai_client.chat.completions.create(  # type: ignore[union-attr]
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1,
                ),
                timeout=timeout_s,
            )
            content = (
                getattr(getattr(raw.choices[0], "message", {}), "content", "") or ""
            )
        # -----------------------------------------------------------

        # TODO [L3]: Parse LLM response for conflict detection - AIKH-488/AIKH-603 (TC-CONFLICT-003)
        # MCP Tool: detect_document_conflicts
        # Use Case: Extract conflict_detected, confidence, and explanation from LLM response
        # Business Rule: Response format is "CONFLICT_DETECTED|CONFIDENCE|EXPLANATION"
        #   - conflict_token: truthy (yes/true/y/1) or falsy (no/false/n/0)
        #   - confidence: float 0.0-1.0 (clamped)
        #   - explanation: optional text description
        # Test: test_llm_validation_tier
        # -----------------------------------------------------------
        content = (content or "").strip()
        parts = content.split("|", 2)
        if len(parts) < 2:
            return False, "Invalid LLM response format", 0.0

        conflict_token = parts[0].strip().lower()
        truthy = {"yes", "true", "y", "1"}
        falsy = {"no", "false", "n", "0"}
        if conflict_token in truthy:
            conflict_detected = True
        elif conflict_token in falsy:
            conflict_detected = False
        else:
            conflict_detected = False

        try:
            confidence_val = float(parts[1].strip())
        except Exception:
            confidence_val = 0.0
        confidence = max(0.0, min(1.0, confidence_val))

        explanation = parts[2].strip() if len(parts) > 2 else ""
        return conflict_detected, explanation or "", confidence
        # -----------------------------------------------------------
    except TimeoutError:
        detector.logger.warning("LLM conflict validation timed out")
        return False, "LLM validation timeout", 0.0
    except Exception as e:  # pragma: no cover
        detector.logger.error(f"Error in LLM conflict validation: {e}")
        return False, f"LLM validation error: {str(e)}", 0.0


async def llm_analyze_conflicts(
    detector: Any, doc1: Any, doc2: Any, similarity_score: float
) -> dict | None:
    """Perform detailed LLM conflict analysis with structured output.

    Use Case: Deep analysis of document conflicts with JSON response parsing
    Data Flow: (doc1, doc2, similarity_score) -> LLM with system prompt -> parse JSON
    Business Rule: Returns structured conflict data with type, confidence, explanation
    Git: "Enhanced LLM conflict analysis" - Structured conflict responses
    """
    # TODO [L3]: Check LLM provider for detailed analysis - AIKH-488/AIKH-603 (TC-CONFLICT-003)
    # MCP Tool: detect_document_conflicts
    # Use Case: Validate LLM provider availability for deep conflict analysis
    # Business Rule: Return None if no provider available (graceful degradation)
    # Test: test_llm_validation_tier
    # -----------------------------------------------------------
    provider = getattr(getattr(detector, "engine", None), "llm_provider", None)
    openai_client = getattr(detector, "openai_client", None)
    if provider is None and openai_client is None:
        return None
    # -----------------------------------------------------------

    try:
        # TODO [L3]: Execute detailed LLM conflict analysis - AIKH-488/AIKH-603 (TC-CONFLICT-003)
        # MCP Tool: detect_document_conflicts
        # Use Case: Call LLM with system prompt for structured conflict analysis
        # Business Rule: Use "conflict detection assistant" system prompt for consistent responses
        # Data Flow: (doc1.text, doc2.text) -> LLM -> JSON response with conflicts array
        # Test: test_llm_validation_tier
        # -----------------------------------------------------------
        if provider is not None:
            chat_client = provider.chat()
            response = await chat_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a conflict detection assistant.",
                    },
                    {
                        "role": "user",
                        "content": f"Analyze conflicts between:\nDoc1: {doc1.text}\nDoc2: {doc2.text}",
                    },
                ],
                model=(
                    getattr(getattr(detector, "_settings", {}), "get", lambda *_: None)(
                        "conflict_llm_model", None
                    )
                    or "gpt-3.5-turbo"
                ),
                max_tokens=500,
                temperature=0.1,
            )
            content = (response or {}).get("text", "")
        else:
            raw = await openai_client.chat.completions.create(  # type: ignore[union-attr]
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a conflict detection assistant.",
                    },
                    {
                        "role": "user",
                        "content": f"Analyze conflicts between:\nDoc1: {doc1.text}\nDoc2: {doc2.text}",
                    },
                ],
                max_tokens=500,
                temperature=0.1,
            )
            content = (
                getattr(getattr(raw.choices[0], "message", {}), "content", "") or ""
            )
        # -----------------------------------------------------------

        import json

        # 'content' computed above for either provider path

        # TODO [L3]: Extract JSON from LLM response - AIKH-488/AIKH-603 (TC-CONFLICT-003)
        # MCP Tool: detect_document_conflicts
        # Use Case: Parse JSON object from LLM response which may contain extra text
        # Business Rule: Find first valid JSON object using balanced brace matching
        # Data Flow: raw text -> find "{" ... "}" -> extract JSON substring
        # Test: test_llm_validation_tier
        # -----------------------------------------------------------
        def extract_json_object(text: str, max_scan: int | None = None) -> str | None:
            if not text:
                return None
            n = len(text)
            limit = (
                min(n, max_scan) if isinstance(max_scan, int) and max_scan > 0 else n
            )
            start = text.find("{", 0, limit)
            if start == -1:
                return None
            in_string = False
            escape = False
            depth = 0
            i = start
            while i < limit:
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    else:
                        if ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                    i += 1
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        return text[start : end + 1]
                i += 1
            return None

        llm_result: dict | None
        try:
            llm_result = json.loads(content)
        except Exception:
            extracted = extract_json_object(content)
            if extracted is None:
                detector.logger.warning(
                    "No JSON object found in LLM content", snippet=content[:200]
                )
                return None
            try:
                llm_result = json.loads(extracted)
            except Exception as json_err:  # pragma: no cover
                detector.logger.warning(
                    "Failed to parse extracted JSON from LLM content",
                    error=str(json_err),
                    snippet=extracted[:200],
                )
                return None

        if not isinstance(llm_result, dict):
            return None

        # TODO [L3]: Process LLM conflict analysis result - AIKH-488/AIKH-603 (TC-CONFLICT-003)
        # MCP Tool: detect_document_conflicts
        # Use Case: Extract structured conflict data from parsed JSON response
        # Business Rule: Normalize has_conflicts (bool/int/string) and extract:
        #   - conflicts: list of conflict objects with type
        #   - confidence: float 0.0-1.0 (default 0.5 if missing)
        #   - explanation: string description of conflict
        # Test: test_llm_validation_tier
        # -----------------------------------------------------------
        raw_has_conflicts = llm_result.get("has_conflicts", False)
        if isinstance(raw_has_conflicts, bool):
            has_conflicts = raw_has_conflicts
        elif isinstance(raw_has_conflicts, int | float):
            has_conflicts = bool(raw_has_conflicts)
        else:
            has_conflicts = str(raw_has_conflicts).strip().lower() in {
                "true",
                "yes",
                "1",
            }

        if not has_conflicts:
            return None

        conflicts = llm_result.get("conflicts")
        if not isinstance(conflicts, list):
            conflicts = []

        conflict_type = "unknown"
        if conflicts and isinstance(conflicts[0], dict):
            conflict_type = conflicts[0].get("type", "unknown")

        raw_conf = llm_result.get("confidence")
        if raw_conf is None and conflicts and isinstance(conflicts[0], dict):
            raw_conf = conflicts[0].get("confidence")
        try:
            confidence = float(raw_conf) if raw_conf is not None else 0.5
        except Exception:
            try:
                confidence = float(str(raw_conf))
            except Exception:
                confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        explanation = llm_result.get("explanation")
        if not isinstance(explanation, str):
            explanation = "LLM analysis"

        return {
            "conflicts": conflicts,
            "has_conflicts": True,
            "confidence": confidence,
            "explanation": explanation,
            "similarity_score": similarity_score,
            "type": conflict_type,
        }
        # -----------------------------------------------------------
    except Exception as e:  # pragma: no cover
        detector.logger.warning("LLM conflict analysis failed", error=str(e))
        return None
