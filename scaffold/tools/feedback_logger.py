#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

def main():
    p = argparse.ArgumentParser(description='Append a JSONL feedback record.')
    p.add_argument('--log', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--payload-json', default='-', help="JSON object or '-' for stdin")
    p.add_argument('--notes', default='')
    a = p.parse_args()
    raw = sys.stdin.read() if a.payload_json == '-' else a.payload_json
    rec = {'timestamp': utc_now(), 'run_id': a.run_id, 'source': a.source, 'payload': json.loads(raw or '{}'), 'notes': a.notes}
    out = Path(a.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a', encoding='utf-8') as h:
        h.write(json.dumps(rec, sort_keys=True) + chr(10))
    print(json.dumps(rec, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
