from unittest.mock import patch

from propose_round2_splits import propose


def test_split_proposal_keeps_pilot_and_near_duplicate_groups_together():
    frozen=[dict(image_id='p1',image_path='p1.jpg',source_group='pilot_train',split='train'),
            dict(image_id='p2',image_path='p2.jpg',source_group='pilot_val',split='val'),
            dict(image_id='p3',image_path='p3.jpg',source_group='pilot_test',split='test')]
    extra=[dict(image_id=f'n{i}',image_path=f'n{i}.jpg',source_group=f'new{i}',split='val') for i in range(10)]
    original=frozen+extra
    def fingerprint(path):
        name=path.name
        index=int(name[1:-4]) if name.startswith('n') else 100+int(name[1:-4])
        # identical near hash for n0/n1, distinct codewords for others
        return name, 0 if index in (0,1) else int('1'*8,2) << (index*8), (100,200,3)
    with patch('dataset_audit.fingerprint',side_effect=fingerprint):
        result=propose(original,frozen)
    assert result[:3]==frozen
    assert original[3]['split']=='val'
    assert result[3]['split']==result[4]['split']