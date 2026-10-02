"""Read-only conservative submission count and bucket-space gate.

The organizer does not state a day-reset timezone. Requiring fewer than five
existing submissions in the preceding 24 hours also respects a five-per-day
calendar limit in any timezone. This script never uploads or deletes objects.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess


def check(mc: Path, remote: str, submission: Path) -> dict:
    completed = subprocess.run([str(mc.resolve()), 'ls', remote, '--json', '--no-color'],
                               check=True, capture_output=True, text=True)
    objects = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if any(obj.get('status') != 'success' or obj.get('type') != 'file' for obj in objects):
        raise ValueError('Bucket listing is incomplete or contains unexpected entries')
    now = datetime.now(timezone.utc)
    prior = [obj for obj in objects if obj['key'].endswith('.parquet')]
    recent = [obj for obj in prior if datetime.fromisoformat(obj['lastModified'].replace('Z','+00:00'))
              > now-timedelta(hours=24)]
    existing = {obj['key'] for obj in objects}
    match = re.fullmatch(r'(.+)_v([1-9][0-9]*)\.parquet', submission.name)
    if not match or not submission.is_file():
        raise ValueError('Expected a finalized <team>_vN.parquet file')
    team, version = match.group(1), int(match.group(2))
    if remote.rstrip('/').rsplit('/',1)[-1] != f'prc-2026-{team}':
        raise ValueError('Submission team and destination bucket disagree')
    versions = [int(m.group(1)) for obj in prior
                if (m := re.fullmatch(re.escape(team)+r'_v([1-9][0-9]*)\.parquet', obj['key']))]
    current_bytes = sum(int(obj['size']) for obj in objects)
    projected = current_bytes+submission.stat().st_size
    report = {'checked_at_utc': now.isoformat(), 'policy': 'At most five submissions in any rolling 24 hours',
              'existing_submission_count_24h': len(recent),
              'remaining_slots_before_upload': max(0,5-len(recent)),
              'bucket_bytes': current_bytes, 'projected_bucket_bytes': projected,
              'bucket_limit_bytes_conservative': 1_000_000_000,
              'submission': submission.name, 'version': version,
              'version_is_new': submission.name not in existing and version>max(versions, default=0)}
    report['allowed'] = len(recent)<5 and projected<=1_000_000_000 and report['version_is_new']
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mc',type=Path,default=Path('artifacts/tools/mc-community.exe'))
    p.add_argument('--remote',default='opensky/prc-2026-merry-mushroom/')
    p.add_argument('--submission',type=Path,required=True)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    report=check(args.mc,args.remote,args.submission)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['allowed']:
        raise SystemExit('Submission blocked by count, bucket size or version gate')


if __name__=='__main__':
    main()
