import json
from collections import Counter
from pathlib import Path


def test_frozen_options_direction_counts_are_explicit():
    root=Path(__file__).resolve().parents[1]
    changes=json.loads((root/'reports/split_options_20261124T152525043845Z/options.json').read_text(encoding='utf-8'))['option_B']['changes']
    assert Counter((c['from'],c['proposed_to']) for c in changes)=={('val','train'):31,('train','val'):2}