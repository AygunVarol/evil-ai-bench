#!/usr/bin/env python3
"""Validate Evil-AI Bench JSON suites without running model APIs.

The public repository ships redacted suite metadata. When redacted files are
present, this script validates the published counts and reports that full
execution requires the restricted research corpus.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {"smart_home", "healthcare_iot", "industrial_ics", "public_infrastructure", "smart_building"}
ATTACK_SUITES = {
    "persuasion": ROOT / "attacks" / "persuasion_attacks.json",
    "unsafe_action": ROOT / "attacks" / "unsafe_action_attacks.json",
    "data_leak": ROOT / "attacks" / "data_leak_attacks.json",
    "mitm": ROOT / "attacks" / "mitm_attacks.json",
    "prompt_injection": ROOT / "attacks" / "prompt_injection_attacks.json",
}
CAPABILITY_SUITE = ROOT / "agentic-capability" / "capability_check_attacks.json"
BENIGN_SUITE = ROOT / "benign" / "benign_requests.json"
CALL_RE = re.compile(r'^CALL_TOOL name="[^"]+" args=\{.*\}$')


def load(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_redacted_suite(path: Path, expected_count: int, errors: List[str]) -> bool:
    payload = load(path)
    if not payload.get("redacted"):
        return False

    if payload.get("scenario_count") != expected_count:
        errors.append(
            f"{path}: redacted scenario_count is {payload.get('scenario_count')!r}, "
            f"expected {expected_count}"
        )
    domains = set(payload.get("domains", []))
    if domains != DOMAINS:
        errors.append(f"{path}: redacted domains do not match expected domain set")
    if not payload.get("public_release_note"):
        errors.append(f"{path}: redacted suite missing public_release_note")
    return True


def validate_common(path: Path, expected_category: str, expected_count: int, errors: List[str]) -> List[Dict]:
    if is_redacted_suite(path, expected_count, errors):
        return []

    payload = load(path)
    if payload.get("category") != expected_category:
        errors.append(f"{path}: category is {payload.get('category')!r}, expected {expected_category!r}")

    records = payload.get("attacks")
    if not isinstance(records, list):
        errors.append(f"{path}: attacks must be a list")
        return []
    if len(records) != expected_count:
        errors.append(f"{path}: expected {expected_count} records, found {len(records)}")

    seen = set()
    domain_counts = {domain: 0 for domain in DOMAINS}
    for index, record in enumerate(records, start=1):
        record_id = record.get("id")
        if not record_id:
            errors.append(f"{path}: record {index} is missing id")
        elif record_id in seen:
            errors.append(f"{path}: duplicate id {record_id}")
        seen.add(record_id)

        domain = record.get("domain")
        if domain not in DOMAINS:
            errors.append(f"{path}:{record_id}: invalid domain {domain!r}")
        else:
            domain_counts[domain] += 1

        for field in ("prompt", "risk_type", "expected_safe_behavior"):
            if not record.get(field):
                errors.append(f"{path}:{record_id}: missing {field}")

        tool = record.get("tool")
        if tool:
            for field in ("name", "description", "call_pattern"):
                if not tool.get(field):
                    errors.append(f"{path}:{record_id}: tool missing {field}")
            if tool.get("call_pattern") and not CALL_RE.match(tool["call_pattern"]):
                errors.append(f"{path}:{record_id}: invalid CALL_TOOL pattern")

    for domain in DOMAINS:
        if domain_counts[domain] != 10:
            errors.append(f"{path}: expected 10 records for {domain}, found {domain_counts[domain]}")

    return records


def validate_persuasion(records: List[Dict], path: Path, errors: List[str]) -> None:
    for record in records:
        followups = record.get("followups")
        if not isinstance(followups, list) or len(followups) != 3:
            errors.append(f"{path}:{record.get('id')}: persuasion record must have exactly 3 followups")
        elif any(not isinstance(item, str) or not item.strip() for item in followups):
            errors.append(f"{path}:{record.get('id')}: followups must be non-empty strings")


def validate_benign(path: Path, errors: List[str]) -> None:
    records = validate_common(path, "benign", 50, errors)
    for record in records:
        if "tool_required" not in record:
            errors.append(f"{path}:{record.get('id')}: benign record missing tool_required")
        if record.get("tool_required") and not record.get("tool"):
            errors.append(f"{path}:{record.get('id')}: tool_required is true but no tool is defined")


def main() -> int:
    errors: List[str] = []
    redacted_count = 0

    for category, path in ATTACK_SUITES.items():
        if is_redacted_suite(path, 50, errors):
            redacted_count += 1
            continue
        records = validate_common(path, category, 50, errors)
        if category == "persuasion":
            validate_persuasion(records, path, errors)

    if is_redacted_suite(CAPABILITY_SUITE, 50, errors):
        redacted_count += 1
    else:
        validate_common(CAPABILITY_SUITE, "capability_check", 50, errors)

    if is_redacted_suite(BENIGN_SUITE, 50, errors):
        redacted_count += 1
    else:
        validate_benign(BENIGN_SUITE, errors)

    if errors:
        print("Suite validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    if redacted_count:
        print("Redacted suite metadata validation passed.")
        print("  - 50 scenarios per adversarial category")
        print("  - 50 capability checks")
        print("  - 50 benign false-positive checks")
        print("  - Full execution requires the restricted research corpus")
    else:
        print("Suite validation passed.")
        print("  - 50 scenarios per adversarial category")
        print("  - 50 capability checks")
        print("  - 50 benign false-positive checks")
        print("  - 3 follow-ups for every persuasion scenario")
    return 0


if __name__ == "__main__":
    sys.exit(main())
