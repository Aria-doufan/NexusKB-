from __future__ import annotations

import re

from app.schemas.rag import MetadataFilterDecision

ALLOWED_SOURCE_TYPES = {
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
}

ALLOWED_DOC_SEMANTIC_TYPES = {
    "account_notes",
    "chat_thread",
    "code_change",
    "email_thread",
    "faq",
    "generic_doc",
    "issue_ticket",
    "meeting_notes",
    "playbook",
    "policy_rule",
    "technical_doc",
}

SOURCE_PATTERNS = {
    "slack": re.compile(r"\bslack\b|聊天|群里|频道", re.IGNORECASE),
    "gmail": re.compile(r"\bgmail\b|\bemail\b|邮件|邮箱", re.IGNORECASE),
    "linear": re.compile(r"\blinear\b", re.IGNORECASE),
    "google_drive": re.compile(r"\bgoogle drive\b|drive 文档|共享文档", re.IGNORECASE),
    "hubspot": re.compile(r"\bhubspot\b|客户记录|\baccount notes\b", re.IGNORECASE),
    "fireflies": re.compile(r"\bfireflies\b|会议转录|\bmeeting transcript\b", re.IGNORECASE),
    "github": re.compile(r"\bgithub\b|\bpull request\b|\bpr\b|\bcommit\b|\brepo\b|代码", re.IGNORECASE),
    "jira": re.compile(r"\bjira\b|\bticket\b|\bissue\b|\bbug\b|缺陷", re.IGNORECASE),
    "confluence": re.compile(r"\bconfluence\b|\bwiki\b|知识库页面", re.IGNORECASE),
}

SEMANTIC_PATTERNS = {
    "policy_rule": re.compile(r"\bpolicy\b|\brule\b|\bsop\b|规定|政策|制度|合规|报销|\bpto\b", re.IGNORECASE),
    "playbook": re.compile(r"\bplaybook\b|\brunbook\b|操作手册|应急手册|流程手册", re.IGNORECASE),
    "meeting_notes": re.compile(r"\bmeeting notes\b|会议记录|纪要|\bmeeting\b", re.IGNORECASE),
    "email_thread": re.compile(r"\bemail thread\b|邮件线程|邮件", re.IGNORECASE),
    "chat_thread": re.compile(r"\bchat thread\b|聊天记录|\bslack thread\b|讨论串", re.IGNORECASE),
    "issue_ticket": re.compile(r"\bticket\b|\bissue\b|\bbug\b|\bincident\b|工单|缺陷|事故", re.IGNORECASE),
    "code_change": re.compile(r"\bpull request\b|\bpr\b|\bcommit\b|\bdiff\b|代码变更|\brepo\b", re.IGNORECASE),
    "account_notes": re.compile(r"\baccount notes\b|客户记录|客户备注|\bcrm\b", re.IGNORECASE),
    "faq": re.compile(r"\bfaq\b|\bfrequently asked questions\b|常见问题", re.IGNORECASE),
    "technical_doc": re.compile(
        r"\brfc\b|\badr\b|\btechnical spec\b|\barchitecture\b|\bdesign doc\b|架构|技术方案",
        re.IGNORECASE,
    ),
}

BROAD_INTENTS = {"multi_hop", "comparison", "completeness", "conflicting_info"}


def _normalize_metadata_value(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _dedupe_allowed(values: list[str], allowed: set[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_metadata_value(value)
        if normalized in allowed and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def plan_metadata_filter(query: str, rag_intent: str, source_hints: list[str] | None = None) -> MetadataFilterDecision:
    query_text = query or ""
    sources = [source for source, pattern in SOURCE_PATTERNS.items() if pattern.search(query_text)]
    semantic_types = [semantic_type for semantic_type, pattern in SEMANTIC_PATTERNS.items() if pattern.search(query_text)]
    sources.extend(source_hints or [])
    sources = _dedupe_allowed(sources, ALLOWED_SOURCE_TYPES)
    semantic_types = _dedupe_allowed(semantic_types, ALLOWED_DOC_SEMANTIC_TYPES)

    if not sources and not semantic_types:
        return MetadataFilterDecision(mode="none", confidence=0.0, reason="No whitelisted metadata constraint detected.")

    explicit_source = any(pattern.search(query_text) for pattern in SOURCE_PATTERNS.values())
    explicit_semantic = any(pattern.search(query_text) for pattern in SEMANTIC_PATTERNS.values())
    if rag_intent in BROAD_INTENTS:
        return MetadataFilterDecision(
            mode="soft",
            source_types=sources,
            doc_semantic_types=semantic_types,
            confidence=0.58,
            reason="Broad intent may need cross-source evidence; use metadata as a boost instead of a hard filter.",
        )

    if explicit_source and explicit_semantic:
        return MetadataFilterDecision(
            mode="hard",
            source_types=sources,
            doc_semantic_types=semantic_types,
            confidence=0.9,
            reason="Query explicitly names both source and document type constraints.",
        )

    return MetadataFilterDecision(
        mode="soft",
        source_types=sources,
        doc_semantic_types=semantic_types,
        confidence=0.65,
        reason="Query suggests metadata constraints but does not require a narrow hard filter.",
    )
