from prepare_annotation_round import build_queue


def test_queue_never_labels_and_groups_duplicates():
    images=[dict(image_path=f'dataset/{i}.jpg',readable=True,sha256=str(i),width=600,height=450) for i in range(12)]
    audit=dict(images=images,exact_duplicate_groups=[['dataset/0.jpg','dataset/1.jpg']],near_duplicate_candidates=[])
    rows=build_queue(audit,10)
    assert len(rows)==10
    assert not {'dataset/0.jpg','dataset/1.jpg'}<={r['image_path'] for r in rows}
    assert all(r['review_status']=='unlabeled' and r['split']=='UNASSIGNED' for r in rows)