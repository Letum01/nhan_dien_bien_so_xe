import json

import pytest

from annotation_store import AnnotationStore, canvas_boxes
from dataset_validation import validate_boxes, group_split_plan


def test_immutable_history_and_concurrent_edit(tmp_path):
    store=AnnotationStore(tmp_path/"labels.sqlite3")
    row={"image_id":"abc","image_path":"x.jpg","review_status":"proposed"}
    first=store.save(row,0)
    assert first["annotation_version"]==1
    with pytest.raises(ValueError,match="confirmation"):
        store.save({**row,"review_status":"reviewed","source_group":"car"})
    second=store.save({**row,"review_status":"needs_review"},1)
    assert len(store.history("abc"))==2
    with pytest.raises(ValueError,match="another session"):
        store.save(row,1)
    assert store.latest("abc")["annotation_version"]==2


def test_canvas_coordinate_conversion():
    objects=[dict(type="rect",left=20,top=40,width=100,height=80,scaleX=1.5,scaleY=1)]
    assert canvas_boxes(objects,.5,600,450)==[[40,80,300,160]]
    with pytest.raises(ValueError,match="outside"):
        canvas_boxes(objects,.1,600,450)


def test_box_validation_rejects_nan_and_duplicates():
    with pytest.raises(ValueError):validate_boxes([[0,0,float("nan"),10]],100,100)
    with pytest.raises(ValueError,match="Duplicate"):
        validate_boxes([[0,0,20,20],[0,0,20,20]],100,100)


def test_split_plan_deterministic_source_groups():
    rows=[{"image_path":str(i),"source_group":f"g{i//2}"} for i in range(40)]
    first=group_split_plan(rows)
    assert first==group_split_plan(rows)
    assert len(first["groups"])==20
    assert set(first["groups"].values())=={"train","val","test"}


def test_reviewed_glyph_export_preserves_original(tmp_path):
    import cv2
    import numpy as np
    from label_characters import export_reviewed_glyphs
    source=tmp_path/"crop.png"
    image=np.full((40,60,3),240,np.uint8)
    cv2.putText(image,"A",(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.imwrite(str(source),image)
    before=source.read_bytes()
    regions=[{"bbox":[3,5,25,30],"label":"A","line":0,"order":0,"confirmed":True}]
    export_reviewed_glyphs(source,regions,tmp_path/"out","vehicle1","train")
    assert source.read_bytes()==before
    assert cv2.imread(str(tmp_path/"out/000.png"),0).shape==(32,32)
    assert (tmp_path/"out/originals/000.png").exists()
    assert (tmp_path/"out/characters_to_review.csv").exists()


def test_unreviewed_glyphs_never_export(tmp_path):
    import cv2
    import numpy as np
    from label_characters import export_reviewed_glyphs
    source=tmp_path/"crop.png";cv2.imwrite(str(source),np.zeros((40,60,3),np.uint8))
    with pytest.raises(ValueError,match="human confirmation"):
        export_reviewed_glyphs(source,[{"bbox":[3,5,25,30],"label":"A","confirmed":False}],tmp_path/"out","g","train")
    assert not (tmp_path/"out").exists()