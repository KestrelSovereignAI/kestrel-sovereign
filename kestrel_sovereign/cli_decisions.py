"""``kestrel decisions {models,eval}`` — decision models (#3424, spec §9).

``models`` shows what each decision-capable route serves right now (after a
fresh discovery, including pin canaries). ``eval`` runs a labelled sample set
through ``LLMService.decide`` against candidate models and prints calibration
metrics plus a ``[decisions.thresholds...]`` snippet to accept.

Privacy: sample files outside the package's shipped synthetic sets may hold
real memory or conversation text, so they are evaluated on local routes only
unless ``--allow-cloud`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from kestrel_sovereign.llm.decisions.evaluation import (
    BASELINES,
    PACKAGED_SAMPLES_DIR,
    SampleError,
    evaluate_baseline,
    evaluate_model,
    load_samples,
    render_report,
    render_threshold_snippet,
    sample_files,
)


def add_decisions_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "decisions", help="Decision models (typed choice/score/noul): list and evaluate"
    )
    sub = parser.add_subparsers(dest="decisions_command")

    models_p = sub.add_parser("models", help="Discover and list decision models per route")
    models_p.add_argument("--json", action="store_true", help="Print JSON instead of text")

    eval_p = sub.add_parser("eval", help="Evaluate decision models on a labelled sample set")
    eval_p.add_argument("--caller", required=True, help="Caller id the thresholds are for")
    eval_p.add_argument(
        "--samples", nargs="+", type=Path, default=None,
        help="Sample files or directories (*.jsonl). Default: the package's "
        "synthetic set for --caller",
    )
    eval_p.add_argument(
        "--route", action="append", default=None,
        help="Only routes matching this <vendor> or <vendor>:<route> (repeatable)",
    )
    eval_p.add_argument(
        "--model", action="append", default=None,
        help="Only models whose id contains this text (repeatable)",
    )
    privacy = eval_p.add_mutually_exclusive_group()
    privacy.add_argument("--local-only", action="store_true", help="Use local routes only")
    privacy.add_argument(
        "--allow-cloud", action="store_true",
        help="Allow cloud routes for sample files outside the shipped synthetic sets",
    )
    eval_p.add_argument("--timeout", type=float, default=30.0, help="Per-request deadline (s)")
    eval_p.add_argument("--concurrency", type=int, default=4, help="Requests in flight per model")
    eval_p.add_argument(
        "--target-accuracy", type=float, default=0.9,
        help="Accuracy a choice/score threshold must reach on the answers it keeps",
    )
    eval_p.add_argument(
        "--baseline", action="append", default=None,
        help="Also score a caller-registered baseline on the same samples "
        "(e.g. memory_answerability: chat)",
    )
    eval_p.add_argument("--json", type=Path, default=None, help="Also write the report as JSON")


def _is_packaged(path: Path) -> bool:
    try:
        path.resolve().relative_to(PACKAGED_SAMPLES_DIR.resolve())
        return True
    except ValueError:
        return False


def candidate_selectors(
    routes: Sequence[Dict[str, Any]],
    *,
    route_filters: Optional[Sequence[str]],
    model_filters: Optional[Sequence[str]],
    local_only: bool,
) -> List[str]:
    """``<vendor>:<route>/<model>`` for every model an eval may ask for.

    A pinned route offers only its verified pin (``decide`` refuses any other
    model on it); an unpinned route offers every discovered model.
    """

    selectors: List[str] = []
    for route in routes:
        name = route["route"]
        if local_only and not route["is_local"]:
            continue
        if route_filters and not any(
            name == f or route["vendor"] == f for f in route_filters
        ):
            continue
        if route["pin"] is not None:
            ids = [route["pin"]] if route["pin_status"] == "verified" else []
        else:
            ids = [m["id"] for m in route["models"]]
        for model_id in ids:
            if model_filters and not any(f in model_id for f in model_filters):
                continue
            selectors.append(f"{name}/{model_id}")
    return selectors


def _print_models(routes: Sequence[Dict[str, Any]]) -> None:
    if not routes:
        print("No decision-capable routes are configured.")
        return
    for route in routes:
        where = "local" if route["is_local"] else "cloud"
        flags = []
        if route["discovery_stale_since"] is not None:
            flags.append("discovery stale")
        if route["pin"] is not None:
            pin = f"pin {route['pin']} ({route['pin_status']}"
            pin += f": {route['pin_reason']})" if route["pin_reason"] else ")"
            flags.append(pin)
        print(f"{route['route']} [{where}]" + (f"  {'; '.join(flags)}" if flags else ""))
        if not route["models"]:
            print("  (no decision models discovered)")
        for model in route["models"]:
            limit = model["context_limit"] if model["context_limit"] is not None else "?"
            print(f"  {model['id']}  context={limit}")


async def _models(service: Any, args: argparse.Namespace) -> int:
    await service.reconcile_decision_capabilities(use_cache=False)
    routes = service.describe_decision_routes()
    if args.json:
        print(json.dumps(routes, indent=2, default=str))
    else:
        _print_models(routes)
    return 0


async def _eval(service: Any, args: argparse.Namespace) -> int:
    if args.samples is None and not (PACKAGED_SAMPLES_DIR / args.caller).is_dir():
        print(
            f"ERROR: no shipped sample set for caller {args.caller!r}; pass --samples",
            file=sys.stderr,
        )
        return 2
    paths = args.samples or [PACKAGED_SAMPLES_DIR / args.caller]
    try:
        files = sample_files(paths)
        samples, digest = load_samples(files)
    except SampleError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    private = [f for f in files if not _is_packaged(f)]
    local_only = args.local_only or (bool(private) and not args.allow_cloud)
    if private and local_only and not args.local_only:
        print(
            "Note: sample files outside the shipped synthetic sets are evaluated "
            "on local routes only (pass --allow-cloud to include cloud routes).",
            file=sys.stderr,
        )

    await service.reconcile_decision_capabilities(use_cache=False)
    selectors = candidate_selectors(
        service.describe_decision_routes(),
        route_filters=args.route,
        model_filters=args.model,
        local_only=local_only,
    )
    if not selectors:
        print("ERROR: no decision model matches; see `kestrel decisions models`", file=sys.stderr)
        return 2

    for name in args.baseline or ():
        if name not in (BASELINES.get(args.caller) or {}):
            print(f"ERROR: caller {args.caller!r} has no baseline {name!r}", file=sys.stderr)
            return 2

    if args.baseline:
        # Chat baselines go through the service's chat paths, some of which
        # (the response audit) do not warm model discovery themselves; a fresh
        # CLI service would otherwise skip every ``model = "auto"`` route. A
        # local-only run warms local routes only, contacting no cloud vendor.
        if local_only:
            await service._ensure_models_discovered(force_local_only=True)
        else:
            await service.discover_all_models()

    reports = []
    for name in args.baseline or ():
        print(f"evaluating baseline {name} on {len(samples)} samples...", file=sys.stderr)
        reports.append(await evaluate_baseline(
            service, args.caller, name, samples,
            local_only=local_only,
            timeout_seconds=args.timeout,
            concurrency=args.concurrency,
            target_accuracy=args.target_accuracy,
        ))
    for selector in selectors:
        print(f"evaluating {selector} on {len(samples)} samples...", file=sys.stderr)
        reports.append(await evaluate_model(
            service, selector, samples,
            caller=args.caller,
            local_only=local_only,
            timeout_seconds=args.timeout,
            concurrency=args.concurrency,
            target_accuracy=args.target_accuracy,
        ))

    print(render_report(reports, samples=len(samples), sample_hash=digest))
    snippet = render_threshold_snippet(
        reports, caller=args.caller, samples=len(samples), sample_hash=digest
    )
    if snippet:
        print("To accept these thresholds, add to kestrel.toml:\n")
        print(snippet)
    if args.json is not None:
        args.json.write_text(json.dumps({
            "caller": args.caller,
            "samples": len(samples),
            "sample_sha256": digest,
            "local_only": local_only,
            "models": [asdict(r) for r in reports],
        }, indent=2, default=str))
    return 0


def run(args: argparse.Namespace) -> int:
    command = getattr(args, "decisions_command", None)
    if command not in {"models", "eval"}:
        print("usage: kestrel decisions {models|eval} ...", file=sys.stderr)
        return 2
    if command == "eval" and args.concurrency < 1:
        print("ERROR: --concurrency must be at least 1", file=sys.stderr)
        return 2

    from kestrel_sovereign.paths import load_project_env, project_dir

    load_project_env(project_dir())

    async def _runner() -> int:
        from kestrel_sovereign.llm.service import LLMService

        service = LLMService()
        try:
            if command == "models":
                return await _models(service, args)
            return await _eval(service, args)
        finally:
            await service.close()

    return asyncio.run(_runner())
