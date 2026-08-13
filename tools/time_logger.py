#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

def main():
    p = argparse.ArgumentParser(description='Append a JSONL time event.')
    p.add_argument('--log', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--phase', required=True)
    p.add_argument('--event', required=True, choices=['start', 'stop', 'mark'])
    p.add_argument('--notes', default='')
    a = p.parse_args()
    out = Path(a.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'timestamp': utc_now(), 'run_id': a.run_id, 'phase': a.phase, 'event': a.event, 'notes': a.notes}
    with out.open('a', encoding='utf-8') as h:
        h.write(json.dumps(payload, sort_keys=True) + chr(10))
    print(json.dumps(payload, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
