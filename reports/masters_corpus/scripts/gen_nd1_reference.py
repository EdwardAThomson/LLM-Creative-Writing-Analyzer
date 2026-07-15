"""Build the nd1 masters reference distribution from the 26 scored sidecars.

Loads every *.nd.json in the DeepSeek nd1 run, aggregates them via
benchmarks.narrative_dynamics.reference.make_reference (per-metric
min/median/max/mean/std across the corpus), stamps judge/book-count/date
metadata, and writes reports/masters_corpus/nd1_reference.json.

Pure local aggregation: no judge calls. Re-run after the sidecars change.
Usage: python scripts/gen_nd1_reference.py [--date YYYY-MM-DD]
"""
import argparse
import glob
import json
import os
import sys

# Make the analyzer package importable when run from anywhere.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from benchmarks.narrative_dynamics.reference import make_reference  # noqa: E402

SIDECAR_DIR = '/home/edward/Projects/StoryDaemon/work/corpus/scores/nd1_ab/deepseek'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nd1_reference.json')
JUDGE = 'ai_helper:openrouter:deepseek/deepseek-chat'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='unstamped',
                    help='ISO date to record in the reference description')
    ap.add_argument('--sidecar-dir', default=SIDECAR_DIR)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.sidecar_dir, '*.nd.json')))
    documents = {}
    judges = set()
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        name = os.path.basename(p)[:-len('.nd.json')]
        documents[name] = d
        if d.get('judge'):
            judges.add(d['judge'])

    if not documents:
        raise SystemExit(f'no *.nd.json found under {args.sidecar_dir}')
    if judges != {JUDGE}:
        # Fail loud rather than silently mixing judges into one reference band.
        raise SystemExit(f'expected a single judge {JUDGE!r}, found {sorted(judges)}')

    description = (
        f'Masters corpus nd1 reference band. n={len(documents)} public-domain '
        f'masterworks, single judge {JUDGE}. Built {args.date}. '
        f'Per-metric min/median/max/mean/std; compare LLM output against this range.'
    )
    ref = make_reference(documents, description=description, benchmark='nd1')

    with open(args.out, 'w') as f:
        json.dump(ref, f, indent=2, sort_keys=True)
        f.write('\n')

    metric_count = sum(len(v) for v in ref['metrics'].values())
    print(f'wrote {os.path.abspath(args.out)}')
    print(f'  books: {len(ref["documents"])}')
    print(f'  metrics families: {list(ref["metrics"].keys())}')
    print(f'  scalar fields summarised: {metric_count}')


if __name__ == '__main__':
    main()
