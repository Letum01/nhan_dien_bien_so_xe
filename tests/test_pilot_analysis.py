from analyze_pilot_confidence import xywh_to_xyxy
from prepare_round2 import prepare_plan


def test_source_box_is_xywh_not_xyxy():
    assert xywh_to_xyxy([168,166,215,170])==[168,166,383,336]


def test_round2_freezes_test_and_does_not_assign_groups_or_splits():
    frozen=[dict(image_id='old',image_path='old.jpg',split='test',source_group='old_group')]
    audit=dict(images=[dict(image_path=p,sha256=p,readable=True,width=600,height=450) for p in ['old.jpg','near.jpg','new.jpg']],
               near_duplicate_candidates=[dict(a='old.jpg',b='near.jpg')],exact_duplicate_groups=[])
    plan=prepare_plan(audit,frozen,candidates=1)
    assert plan['frozen_test_images']==['old.jpg']
    assert [r['image_path'] for r in plan['candidates']]==['new.jpg']
    assert plan['candidates'][0]['split']=='UNASSIGNED'
    assert plan['candidates'][0]['review_status']=='unlabeled'