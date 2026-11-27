from glyph_review_ui import validation_queue


def test_validation_queue_filters_train_and_current_reviews():
    def row(key, split='val', revision=2, text=False):
        return dict(image_id=key,image_path=key+'.jpg',split=split,plate_bbox=[1,2,40,30],
                    detection_revision=revision,source_group=key,text_confirmed=text,plate_index=0)
    entries=[row('train','train'),row('a'),row('b',text=True),row('done')]
    pending,done=validation_queue(entries,[row('a',revision=1),row('done')])
    assert [r['image_id'] for r in pending]==['b','a']
    assert [r['image_id'] for r in done]==['done']