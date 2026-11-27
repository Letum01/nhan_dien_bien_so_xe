from audit_round2_training import quality_flags


def test_tiny_nested_plate_requires_review_without_rewriting():
    row={'boxes':[[218,220,141,109],[292,262,2,2]]}
    flags=quality_flags(row,600,450)
    assert any(f['box_index']==1 and f['reason']=='tiny_plate_candidate_requires_human_inspection' for f in flags)
    assert row['boxes'][1]==[292,262,2,2]


def test_normal_source_xywh_not_interpreted_as_xyxy():
    assert not quality_flags({'boxes':[[168,166,215,170]]},600,480)