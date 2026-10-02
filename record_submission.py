"""Verify downloaded submission bytes and save only a public aggregate receipt.

Raw result files can contain internal evaluation metadata. This command copies
only status, pair count, score and public leaderboard metadata into reports/.
It never reads credentials or uploads anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


COMPETITION = 'bb3693e1-26bc-4a9e-8619-4fe78b4eab0c'
ENDPOINT = f'https://datacomp.opensky-network.org/api/competitions/{COMPETITION}/leaderboard'


def get_page(params):
    with urlopen(ENDPOINT + '?' + urlencode(params), timeout=30) as response:
        page = json.load(response)
    if not isinstance(page.get('items'), list):
        raise ValueError('Unexpected public leaderboard response')
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, type=int)
    parser.add_argument('--team', default='merry-mushroom')
    parser.add_argument('--submission-dir', type=Path, default=Path('submissions'))
    parser.add_argument('--receipt-dir', type=Path, default=Path('artifacts/submission-receipt'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    args = parser.parse_args()
    filename = f'{args.team}_v{args.version}.parquet'
    source = args.submission_dir / filename
    manifest = json.loads(source.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    receipt = json.loads((args.receipt_dir / (filename + '_result.json')).read_text(encoding='utf-8-sig'))
    downloaded = args.receipt_dir / ('verified_' + filename)
    digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
    if digest != manifest['sha256'] or digest != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Remote readback does not match the finalized submission')
    if receipt['status'] != 'Succeeded' or receipt['used_pairs'] != manifest['rows']:
        raise ValueError('The organizer has not accepted every submission pair')

    team_items = []
    params = {'teamName': args.team, 'limit': 200}
    while True:
        page = get_page(params)
        team_items.extend(page['items'])
        if not page.get('nextCursor'):
            break
        params['cursor'] = page['nextCursor']
    matching = [item for item in team_items if item['filename'] == filename]
    if not matching:
        raise ValueError('The accepted submission is not yet on the public leaderboard')
    item = max(matching, key=lambda entry: entry['processedAt'])
    if abs(float(item['score']) - float(receipt['score'])) > 1e-7:
        raise ValueError('The organizer receipt and public leaderboard disagree')
    best_score = min(float(entry['score']) for entry in team_items)
    better = set()
    params = {'limit': 200}
    previous_score = -float('inf')
    while True:
        page = get_page(params)
        reached_cutoff = False
        for entry in page['items']:
            score = float(entry['score'])
            if score < previous_score:
                raise ValueError('The public leaderboard is not sorted by score')
            previous_score = score
            if score >= best_score:
                reached_cutoff = True
                break
            better.add(entry['teamName'])
        if reached_cutoff or not page.get('nextCursor'):
            break
        params['cursor'] = page['nextCursor']

    result = {'team': args.team, 'filename': filename, 'status': receipt['status'],
              'used_pairs': receipt['used_pairs'], 'official_rmse_seconds': float(receipt['score']),
              'processed_at': item['processedAt'], 'sha256': digest,
              'size_bytes': downloaded.stat().st_size, 'remote_readback_sha256_verified': True,
              'team_best_rmse_seconds': best_score, 'team_rank': len(better) + 1,
              'rank_as_of': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
              'team_rank_method': 'Distinct teams with a lower best score, plus one',
              'public_team_results_url': ENDPOINT + '?' + urlencode({'teamName': args.team, 'limit': 200})}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f'submission_v{args.version}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
