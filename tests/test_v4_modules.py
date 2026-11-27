import json
import cv2
import numpy as np

from v4_failure_store import save_case
from v4_geometry import decode, mapped_box, rectify


def test_rectify_preserves_corners_and_maps_boxes():
    crop=np.full((80,160,3),220,np.uint8)
    cv2.rectangle(crop,(12,10),(148,70),(20,20,20),2)
    out,H,reason=rectify(crop,'raw')
    assert reason=='identity' and out.shape==crop.shape
    box=mapped_box([20,15,12,30],H)
    assert box==[20.0,15.0,12.0,30.0]
    # Tilted quadrilateral triggers deskew/perspective without clipping corners.
    tilted=np.full((120,220,3),230,np.uint8)
    pts=np.array([[25,25],[195,12],[200,92],[30,105]])
    cv2.fillPoly(tilted,[pts],(40,40,40))
    cv2.fillPoly(tilted,[pts+np.array([[6,6],[-6,6],[-6,-6],[6,-6]])],(220,220,220))
    warped,M,mode=rectify(tilted,'perspective')
    assert mode=='perspective' and warped.shape[0]>=16 and np.isfinite(M).all()


def test_format_decoder_only_uses_close_top3_evidence():
    classes='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    prob=np.zeros((8,36),np.float32)
    for i,ch in enumerate('65010297'):prob[i,classes.index(ch)]=.95
    prob[2,classes.index('0')]=.52; prob[2,classes.index('D')]=.45
    assert decode(prob,classes)=='65D10297'
    # Large gap must not force a substitution.
    prob[2,classes.index('0')]=.80; prob[2,classes.index('D')]=.15
    assert decode(prob,classes)=='65010297'


def test_failure_case_store_writes_isolated_artifact(tmp_path):
    im=np.zeros((60,100,3),np.uint8); raw=cv2.imencode('.png',im)[1].tobytes()
    res={'plate':{'bbox':[10,10,40,20],'crop_box':[5,5,50,30],'score':.9},'text':{'cnn':'29A12345'}}
    folder=save_case(tmp_path,'sample.png',raw,im,res,'character_yolo',{'cnn':'h'},'29A12346')
    data=json.loads((folder/'case.json').read_text(encoding='utf-8'))
    assert (folder/'crop.png').is_file() and data['user_text_unreviewed']=='29A12346'


def test_self_training_does_not_accept_teacher_agreement_as_provenance():
    from v4_self_training import readiness
    report=readiness([{'status':'AUTO_PSEUDO_LABEL','provenance':'UNKNOWN','source_group':'UNKNOWN'}])
    assert (report['status'],report['filter_passed'],report['unverified'])==('BLOCKED',1,1)