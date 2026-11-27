"""Actual save/reload counter test using synthetic detection/OCR databases."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from annotation_store import AnnotationStore,image_id
from glyph_review import plates_from_detection

ROOT=Path(__file__).resolve().parent


def main():
    validation=os.environ.get('ALPR_OCR_QUEUE')=='validation'
    output=ROOT/'reports'/('glyph_validation_browser' if validation else 'glyph35_counter_browser');output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        folder=Path(temporary);source=folder/'synthetic.jpg'
        image=np.full((100,200,3),230,np.uint8);cv2.putText(image,'A',(30,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3);cv2.imwrite(str(source),image)
        detection=AnnotationStore(folder/'d.db');ocr=AnnotationStore(folder/'o.db')
        detection.save(dict(image_id=image_id(source),image_path=str(source),boxes=[[20,10,100,80]],
                           source_group='SYNTHETIC_FIXTURE',split='val' if validation else 'train',review_status='reviewed',confirmed_by_user=True,
                           plates=[dict(plate_text='A',text_confirmed=True,unreadable=False,layout='one_line')]),0)
        parent=plates_from_detection(detection)[0]
        draft={**parent,'characters':[dict(bbox=[10,10,45,55],label='A',confirmed=True,line=0,order=0)]}
        ocr.save_draft(draft,0)
        env={**os.environ,'ALPR_ANNOTATION_DB':str(detection.path),'ALPR_GLYPH_DB':str(ocr.path),'ALPR_OCR_QUEUE':'validation' if validation else 'all'}
        with (output/'server.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','streamlit','run','label_characters.py','--server.port=8897','--server.headless=true','--browser.gatherUsageStats=false'],cwd=ROOT,env=env,stdout=log,stderr=log)
            try:
                for _ in range(60):
                    try:urllib.request.urlopen('http://127.0.0.1:8897/_stcore/health',timeout=1);break
                    except OSError:time.sleep(.5)
                with sync_playwright() as p:
                    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1500,'height':1600})
                    page.goto('http://127.0.0.1:8897');page.get_by_text('Gắn nhãn OCR / Glyph',exact=True).wait_for(timeout=60000)
                    if validation:
                        page.get_by_role('heading',name='OCR Validation — hoàn thiện').wait_for()
                        page.reload()
                        page.get_by_role('button',name='Lưu Glyph reviewed',exact=True).wait_for(timeout=60000)
                    page.wait_for_timeout(2000)
                    # Confirmation controls deliberately reset on UI load.
                    page.get_by_text('Đã kiểm tra glyph đủ nét',exact=True).click();page.wait_for_timeout(1000)
                    page.get_by_text('Tôi đã kiểm tra từng glyph và thứ tự; xác nhận Glyph reviewed',exact=True).click();page.wait_for_timeout(1000)
                    page.get_by_role('button',name='Lưu Glyph reviewed',exact=True).click();page.wait_for_timeout(2500)
                    def totals():
                        values=[v.strip() for v in page.locator('[data-testid="stMetricValue"]').all_text_contents()]
                        assert values[2:4]==['1','1'],values
                        if validation: assert values[4:7]==['0','1','1'],values
                        assert ocr.latest(parent['image_id'])['review_status']=='reviewed'
                    totals();page.reload();page.wait_for_timeout(2500);totals()
                    assert not page.locator('[data-testid="stException"]').count()
                    page.screenshot(path=str(output/'after_save_reload.png'),full_page=True)
                    browser.close()
                (output/'results.json').write_text(json.dumps({'reviewed_save_then_immediate_counters':'PASS','reload_counters':'PASS',
                    'data':'synthetic temporary databases ONLY','real_labels_modified':False},indent=2),encoding='utf-8')
                print('Save/reload counters Chromium PASS, fixture only')
            finally:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait(timeout=10)


if __name__=='__main__':main()