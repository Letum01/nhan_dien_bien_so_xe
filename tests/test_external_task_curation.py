from pathlib import Path

import cv2
import numpy as np
import pytest

from external_task_curation import classify_tier, glyph_flags
from external_task_train import preflight, digest, write


def test_clean_glyph_is_only_a_candidate_and_kept_without_parent_completeness():
    image=np.full((60,100,3),255,np.uint8)
    cv2.rectangle(image,(24,15),(27,33),(0,0,0),-1)
    info, flags=glyph_flags(image,[20,10,14,28],[])
    assert info['tall_components']==1
    assert classify_tier(flags,[])=='B'
    assert classify_tier(flags,['protected_component'])=='D'


def test_blank_and_invalid_glyphs_are_flagged():
    image=np.full((30,30,3),255,np.uint8)
    assert 'unreliable_foreground_proxy' in glyph_flags(image,[5,5,10,16],[])[1]
    assert glyph_flags(image,[25,5,10,16],[])==(None,['invalid_box'])


def test_actual_train_path_denied_without_approval(tmp_path):
    with pytest.raises(ValueError,match='BLOCKED'):
        preflight(tmp_path,None)


@pytest.mark.parametrize('conflict',[False, True])
def test_frozen_preflight_group_leakage(tmp_path: Path, conflict):
    frozen=[]; glyphs=[]
    for i,split in enumerate(('train','development','test')):
        row=dict(parent=f'{i}.jpg',glyph_index=0,sha256=str(i),character='1')
        glyphs.append({**row,'tier':'B','quarantine':[],'possible_duplicate_component':f'component-{i}'})
        frozen.append({**row,'source_group':str(0 if conflict and i==1 else i),'split':split,'glyph_verified':True})
    def save(name, data):
        path=tmp_path/name; write(path,data);return dict(path=name,sha256=digest(path))
    summary=save('summary.json',{}); contents=save('glyphs.json',glyphs)
    approval=dict(purpose='academic_classifier_training',approved_by='fixture',curation_summary_sha256=summary['sha256'],
                  curation_glyphs_sha256=contents['sha256'],frozen_manifest=save('frozen.json',frozen))
    for field in ('rights_evidence','source_group_evidence','semantic_evidence','leakage_evidence'):
        approval[field]=save(field+'.json',{'fixture_only':True})
    save('approval.json',approval)
    if conflict:
        with pytest.raises(ValueError,match='Cross-split overlap'):
            preflight(tmp_path,tmp_path/'approval.json')
    else:
        rows,_=preflight(tmp_path,tmp_path/'approval.json'); assert len(rows)==3