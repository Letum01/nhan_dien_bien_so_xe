"""Read-only gate: never stage training data from unverified proposals."""
from final_ocr_dataset import ROOT, check_protocol, read


def readiness(rows):
    accepted = [r for r in rows if r.get('status') == 'AUTO_PSEUDO_LABEL']
    unverified = [r for r in accepted if r.get('provenance') != 'VERIFIED_DEVELOPMENT'
                  or not r.get('source_group') or r['source_group'].strip().upper() in ('', 'UNKNOWN', 'UNVERIFIED')]
    return {'status': 'BLOCKED', 'filter_passed': len(accepted), 'unverified': len(unverified),
            'reason': 'No verified source-held-out quality receipt for pseudo geometry and text'}


if __name__ == '__main__':
    import json
    check_protocol()
    print(json.dumps(readiness(read(ROOT/'experiments/alpr_v4/pool_pilot/pseudo_pilot_100.json')['rows']), indent=2))
    raise SystemExit(2)