from pathlib import Path

import cv2
import numpy as np

from evaluate_real_glyph_baseline import draw_sheet,visual_checks


def test_contact_and_visual_screening_preserve_pixels(tmp_path):
    folder=tmp_path/'glyph';folder.mkdir();(folder/'originals').mkdir()
    raw=np.full((40,20,3),240,np.uint8)
    cv2.putText(raw,'1',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.imwrite(str(folder/'originals/000.png'),raw)
    binary=np.zeros((32,32),np.uint8);binary[:,14:18]=255
    cv2.imwrite(str(folder/'000.png'),binary)
    before=(folder/'000.png').read_bytes()
    rows=[dict(image_path='glyph/000.png',class_label='1',split='validation',plate_id='fixture',source_crop='fixture',bbox='[0,0,20,40]')]
    draw_sheet(rows,tmp_path,tmp_path,'contact.png',['I'])
    warnings=visual_checks(rows,tmp_path,tmp_path)
    assert (tmp_path/'contact.png').exists()
    assert (tmp_path/'validation_raw_contact.jpg').exists()
    assert (folder/'000.png').read_bytes()==before
    assert all(r['split']=='validation' for r in warnings)