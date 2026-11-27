from unittest.mock import patch

import numpy as np

from main import end_to_end_pipeline


def test_detected_plate_is_not_hidden_by_segmentation_failure():
    image=np.zeros((150,300,3),np.uint8)
    proposals=[(image[20:60,30:130],(30,20,100,40),.8)]
    trace={}
    with patch('yolo_ui_bridge.detect',return_value=proposals), patch('main.segment_characters',return_value=[]):
        text,_,crop,chars=end_to_end_pipeline(image,'cnn',{'cnn':object()},trace,
                                             detector_type='yolo_isolated',detector_weights='fixture.pt')
    assert text=='UNKNOWN (Segmentation Uncertain)'
    assert crop is not None
    assert trace['detections'][0]['confidence']==.8
    assert trace['status']=='segmentation_uncertain'


def test_empty_yolo_result_never_falls_back_to_opencv():
    image=np.zeros((150,300,3),np.uint8)
    with patch('yolo_ui_bridge.detect',return_value=[]),patch('main.locate_license_plate') as fallback:
        text,_,crop,_=end_to_end_pipeline(image,'cnn',{},detector_type='yolo_isolated',detector_weights='fixture.pt')
    fallback.assert_not_called()
    assert text.startswith('UNKNOWN') and crop is None