import json
import pytest
from ocr_mix_ensemble import evaluate


def test_matching_runs_majority_is_explicit_and_non_destructive(tmp_path):
    paths={}
    for name,labels in [('svm',['A','1']),('rf',['A','2']),('cnn',['B','3'])]:
        paths[name]=tmp_path/(name+'.json')
        paths[name].write_text(json.dumps(dict(dataset_manifest_sha256='fixture',expected=['A','1'],predictions=labels)),encoding='utf-8')
    output=tmp_path/'ensemble.json'
    report=evaluate(paths,output)
    assert report['raw']=='A?' and report['correct']==1
    assert report['uncertain_positions']==[1]
    with pytest.raises(FileExistsError):evaluate(paths,output)


def test_different_manifest_rejected(tmp_path):
    paths={}
    for index,name in enumerate(('svm','rf','cnn')):
        paths[name]=tmp_path/(name+'.json')
        paths[name].write_text(json.dumps(dict(dataset_manifest_sha256=str(index),expected=['A'],predictions=['A'])),encoding='utf-8')
    with pytest.raises(ValueError,match='Different validation'):evaluate(paths,tmp_path/'ensemble.json')