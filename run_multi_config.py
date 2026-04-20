#!/usr/bin/env python3
"""Sequentially run multiple training/evaluation configs.

This helper executes `main.py` multiple times based on a JSON plan file, which
is useful for rebuttal experiments under tight timelines.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _normalize_args(raw_args):
    """Convert args from dict/list/string to a flat list of CLI tokens."""
    if raw_args is None:
        return []

    if isinstance(raw_args, str):
        return shlex.split(raw_args)

    if isinstance(raw_args, list):
        return [str(x) for x in raw_args]

    if isinstance(raw_args, dict):
        tokens = []
        for key, value in raw_args.items():
            if value is None or value is False:
                continue
            flag = key if str(key).startswith('--') else f'--{key}'
            if value is True:
                tokens.append(flag)
            elif isinstance(value, list):
                for item in value:
                    tokens.extend([flag, str(item)])
            else:
                tokens.extend([flag, str(value)])
        return tokens

    raise TypeError(f'Unsupported args type: {type(raw_args)}')


def _load_plan(path: Path) -> Tuple[List[str], List[Dict]]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    common_args = _normalize_args(data.get('common_args', []))
    runs = data.get('runs', [])
    if not runs:
        raise ValueError('Plan file must contain a non-empty "runs" list.')
    return common_args, runs


def _build_command(python_bin: str, main_file: str, common_args: List[str], run_cfg: Dict) -> List[str]:
    run_args = _normalize_args(run_cfg.get('args', []))
    cmd = [python_bin, main_file] + common_args + run_args
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Run multiple configs sequentially.')
    parser.add_argument('--plan', required=True, type=Path,
                        help='Path to JSON plan file containing common_args and runs.')
    parser.add_argument('--python', default=sys.executable,
                        help='Python executable used to launch main.py.')
    parser.add_argument('--main', default='main.py',
                        help='Main training/eval entrypoint file.')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue remaining runs if one run fails.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only print generated commands without executing them.')
    args = parser.parse_args()

    common_args, runs = _load_plan(args.plan)
    results = []

    print(f'[multi-run] loaded {len(runs)} runs from {args.plan}')
    if common_args:
        print(f'[multi-run] common args: {" ".join(common_args)}')

    for idx, run_cfg in enumerate(runs, start=1):
        run_name = run_cfg.get('name', f'run_{idx}')
        cmd = _build_command(args.python, args.main, common_args, run_cfg)
        cmd_str = ' '.join(shlex.quote(x) for x in cmd)

        print(f'\n[{idx}/{len(runs)}] {run_name}')
        print(f'  cmd: {cmd_str}')

        if args.dry_run:
            results.append((run_name, 0, 0.0))
            continue

        start = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - start
        results.append((run_name, proc.returncode, elapsed))

        if proc.returncode != 0:
            print(f'  -> FAILED (code={proc.returncode}, {elapsed:.1f}s)')
            if not args.continue_on_error:
                break
        else:
            print(f'  -> DONE ({elapsed:.1f}s)')

    print('\n=== Summary ===')
    ok = 0
    for run_name, code, elapsed in results:
        status = 'OK' if code == 0 else 'FAIL'
        print(f'{status:>4} | {run_name:<30} | code={code:<3} | {elapsed:7.1f}s')
        if code == 0:
            ok += 1

    if ok != len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
