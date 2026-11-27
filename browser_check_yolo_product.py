"""Actual Streamlit file upload/detector/OCR selection using Playwright."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'reports/round2_streamlit';output.mkdir(exist_ok=True)
    with (output/'server.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-m','streamlit','run','app.py','--server.port=8895',
                               '--server.headless=true','--browser.gatherUsageStats=false'],cwd=ROOT,stdout=log,stderr=log)
        try:
            for _ in range(60):
                try:urllib.request.urlopen('http://127.0.0.1:8895/_stcore/health',timeout=1);break
                except OSError:time.sleep(.5)
            with sync_playwright() as p:
                browser=p.chromium.launch();page=browser.new_page(viewport={'width':1500,'height':1100})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto('http://127.0.0.1:8895');page.get_by_text('Hệ thống nhận diện biển số xe Việt Nam',exact=True).wait_for(timeout=60000)
                page.wait_for_timeout(3000)
                def choose(label,value):
                    controls=page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text(label,exact=True))
                    controls.get_by_role('combobox').click();page.get_by_role('option',name=value,exact=True).click()
                choose('Detector','YOLO vòng 2 — thử nghiệm')
                page.wait_for_timeout(1800)
                page.locator('input[type=file]').set_input_files(str(ROOT/'dataset/license_plate_dataset/0/images/14.jpg'))
                try:
                    page.get_by_text('YOLO detections (xywh / model confidence, không phải OCR confidence):',exact=True).wait_for(timeout=90000)
                except Exception:
                    (output/'failure_page.txt').write_text(page.inner_text('body'),encoding='utf-8')
                    page.screenshot(path=str(output/'failure_page.png'),full_page=True)
                    raise
                operations=[]
                for choice in ('SVM (HOG + Support Vector)','Random Forest (HOG + Ensembles)','CNN (Deep Learning)','Ensemble (Majority — thử nghiệm)'):
                    choose('Chọn mô hình phân loại ký tự:',choice);page.wait_for_timeout(1800)
                    assert not page.locator('[data-testid="stException"]').count()
                    alerts=page.locator('[data-testid="stAlert"]').all_text_contents()
                    assert not any('Lỗi không mong đợi' in a or 'Không thể nhận diện' in a for a in alerts),alerts
                    operations.append({'recognizer':choice,'status':'PASS_no_backend_error'})
                page.get_by_role('button',name='OCR Diagnostics — YOLO vòng 2 (lưu artifact)',exact=True).click()
                page.get_by_text('Không tính accuracy khi chưa có glyph ground truth. Biến thể chỉ thử nghiệm, không đổi production.',exact=True).wait_for(timeout=120000)
                assert not page.locator('[data-testid="stException"]').count()
                assert page.get_by_text('Glyph 0',exact=True).count()>0
                page.screenshot(path=str(output/'yolo_upload.png'),full_page=True)
                assert not errors,errors
                (output/'browser.json').write_text(json.dumps({'actual_file_upload':'PASS','yolo_selection':'PASS',
                    'ocr_switches':operations,'page_errors':errors,'tested_image':'14.jpg validation',
                    'ocr_diagnostics':'PASS rendered glyph0 and baseline artifacts',
                    'ocr_accuracy':'NOT_TESTED','default_detector':'OpenCV not changed'},indent=2),encoding='utf-8')
                browser.close();print('Browser YOLO upload and four OCR choices PASS')
        finally:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait(timeout=10)


if __name__=='__main__':main()