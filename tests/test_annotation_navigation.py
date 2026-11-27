import pytest
from annotation_store import AnnotationStore
from annotation_workflow import split_character, merge_characters, sync_characters


def test_draft_never_overwrites_reviewed_and_cursor_recovers(tmp_path):
    store=AnnotationStore(tmp_path/'a.db')
    row=dict(image_id='one',image_path='a.jpg',review_status='reviewed',source_group='g',confirmed_by_user=True,boxes=[[1,2,30,40]])
    store.save(row,0)
    store.save_draft({**row,'boxes':[]},1)
    store.set_preference('last_source','a.jpg')
    reopened=AnnotationStore(tmp_path/'a.db')
    assert reopened.latest('one')['boxes']==[[1,2,30,40]]
    assert reopened.draft('one')[0]['boxes']==[]
    assert reopened.preference('last_source')=='a.jpg'
    reopened.restore('one',1,1)
    with pytest.raises(ValueError,match='another session'):store.save_draft(row,1)


def test_split_merge_invalidate_labels_and_preserve_rows():
    old=[dict(bbox=[10,5,20,30],label='B',confirmed=True,line=0,order=0)]
    split=split_character(old,0,8)
    assert [c['bbox'] for c in split]==[[10,5,8,30],[18,5,12,30]]
    assert all(not c['confirmed'] and not c['label'] for c in split)
    merged=merge_characters(split,[0,1])
    assert merged[0]['bbox']==old[0]['bbox']
    assert old[0]['confirmed']
    split[1]['line']=1
    with pytest.raises(ValueError,match='rows'):merge_characters(split,[0,1])


def test_resize_invalidates_human_glyph_label():
    chars=[dict(bbox=[10,5,20,30],label='B',confirmed=True,line=0)]
    changed=sync_characters(chars,[[10,5,25,30]])
    assert changed[0]['label']=='' and not changed[0]['confirmed']