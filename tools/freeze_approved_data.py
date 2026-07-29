#!/usr/bin/env python3
"""Public CLI and API for owner-approved real-data evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from approved_data_candidate import (  # noqa: E402
    candidate_value,
    canonical_json_bytes,
    parse_candidate_report,
    parse_domain_signoff,
    signoff_value,
)
from approved_data_evidence import (  # noqa: E402
    calculate_approved_data_binding,
    check_candidate,
    freeze_approved_data,
    manifest_value,
    parse_approved_case_record,
    parse_approved_data_manifest,
    record_value,
    validate_approved_data,
)
from approved_data_model import (  # noqa: E402
    APPROVED_STATUS,
    CASE_IDS,
    CANDIDATE_SCHEMA,
    MANIFEST_SCHEMA,
    RECORD_SCHEMA,
    SIGNOFF_SCHEMA,
    ApprovedCaseIndex,
    ApprovedCaseRecord,
    ApprovedDataBinding,
    ApprovedDataManifest,
    ApprovedRun,
    CandidateCase,
    CandidateReport,
    CanonicalEnvironment,
    DomainSignoff,
    EmbeddedCaseSignoff,
    RelativeFileRecord,
    RepoFileRecord,
    SignedCase,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--domain-signoff", type=Path)
    parser.add_argument("--approved-data-root", type=Path, required=True)
    parser.add_argument("--r22-reference", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-candidate", action="store_true")
    group.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.check_candidate:
        report = check_candidate(
            args.candidate_report,
            args.approved_data_root,
            args.r22_reference,
        )
        print(json.dumps({"status": "PASS", "case_count": len(report.cases)}, sort_keys=True))
        return 0
    if args.domain_signoff is None:
        parser.error("--domain-signoff is required with --output")
    manifest = freeze_approved_data(
        args.candidate_report,
        args.domain_signoff,
        args.approved_data_root,
        args.r22_reference,
        args.output,
    )
    print(json.dumps({"status": "PASS", "case_count": len(manifest.cases)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
