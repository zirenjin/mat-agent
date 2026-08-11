#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
PROTECTED_PARTS = {'raw', 'test', 'ood', 'hidden', 'hidden_test'}
DEFAULT_WRITE_ROOTS = ('playground/runs', 'playground/cache', '.tmp', '/tmp')

def is_protected(path):
    p = Path(path)
    return bool(set(p.parts) & PROTECTED_PARTS) or p.stem in PROTECTED_PARTS
def under_allowed_root(path, repo, roots):
    resolved = (Path(path) if Path(path).is_absolute() else Path(repo) / path).resolve(strict=False)
    for root in roots:
        allowed = (Path(root) if Path(root).is_absolute() else Path(repo) / root).resolve(strict=False)
        try: resolved.relative_to(allowed); return True
        except ValueError: pass
    return False

def main():
    p = argparse.ArgumentParser(description='Check dataset read/write paths against sandbox policy.')
    p.add_argument('path'); p.add_argument('--mode', choices=['read','write'], default='read'); p.add_argument('--repo', default='.')
    p.add_argument('--allowed-write-root', action='append', default=[]); a = p.parse_args(); violations = []
    if a.mode == 'write':
        if is_protected(a.path): violations.append('write_targets_protected_split_or_raw_data')
        if not under_allowed_root(a.path, a.repo, tuple(a.allowed_write_root) or DEFAULT_WRITE_ROOTS): violations.append('write_outside_allowed_roots')
    result = {'ok': not violations, 'mode': a.mode, 'path': a.path, 'violations': violations}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result['ok'] else 2
if __name__ == '__main__': raise SystemExit(main())

