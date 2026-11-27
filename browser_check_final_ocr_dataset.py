"""Exercise manual box/canvas, draft reload and save with an isolated SQLite store."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from annotation_store import AnnotationStore
from glyph_review import ROOT


def main():
    out=ROOT/'reports/final_ocr_browser'; out.mkdir(exist_ok=True)
    with TemporaryDirectory(ignore_cleanup_errors=True) as folder, (out/'server.log').open('w',encoding='utf-8') as log:
        store=AnnotationStore(Path(folder)/'isolated.sqlite3')
        env={**os.environ,'ALPR_FINAL_OCR_DB':str(store.path)}
        url='http://127.0.0.1:8898'
        proc=subprocess.Popen([sys.executable,'-m','streamlit','run','app_ocr_test_dataset.py','--server.port=8898','--server.headless=true'],cwd=ROOT,env=env,stdout=log,stderr=log)
        try:
            for _ in range(60):
                try:
                    with urlopen(url+'/_stcore/health',timeout=1):break
                except OSError:time.sleep(.5)
            with sync_playwright() as p:
                b=p.chromium.launch(); page=b.new_page(viewport={'width':1500,'height':1400});page.goto(url)
                page.get_by_text('OCR Final Test — nguồn độc lập',exact=True).click()
                page.get_by_text('UNKNOWN không phải nguồn độc lập.',exact=False).wait_for()
                from playwright.sync_api import expect
                expect(page.get_by_test_id('stMetricValue')).to_have_text(['0','40','0','0'],timeout=60000)
                page.get_by_text('Mở canvas vẽ box biển',exact=True).click()
                plate_canvas=page.frame_locator('iframe[title*="st_canvas"]').locator('canvas.upper-canvas:visible').last
                plate_canvas.wait_for();plate_canvas.scroll_into_view_if_needed(); pb=plate_canvas.bounding_box()
                page.mouse.move(pb['x']+10,pb['y']+10); page.mouse.down()
                page.mouse.move(pb['x']+120,pb['y']+100,steps=12);page.mouse.up()
                page.wait_for_timeout(800)
                page.get_by_role('button',name='Áp dụng box từ canvas').click()
                page.get_by_text('Crop box thủ công',exact=True).wait_for()
                page.get_by_text('Mở canvas vẽ box biển',exact=True).click()
                canvas=page.frame_locator('iframe[title*="st_canvas"]').last.locator('canvas.upper-canvas:visible').last
                canvas.wait_for(timeout=30000); canvas.scroll_into_view_if_needed(); box=canvas.bounding_box()
                page.mouse.move(box['x']+20,box['y']+20); page.mouse.down();page.mouse.move(box['x']+85,box['y']+115,steps=12);page.mouse.up()
                field=page.get_by_label('Nhãn glyph',exact=True); field.fill('A');field.press('Tab')
                page.get_by_text('Thứ tự hiện tại: A',exact=True).wait_for()
                assert page.get_by_role('button',name='Lưu Glyph reviewed').is_disabled()
                page.reload();page.get_by_text('OCR Final Test — nguồn độc lập',exact=True).click()
                page.get_by_text('Thứ tự hiện tại: A',exact=True).wait_for()
                assert not page.get_by_role('button',name='Tạo glyph proposal chưa nhãn').count()
                page.screenshot(path=out/'manual_draft.png');assert not store.all_latest()
                page.get_by_text('Nhập ảnh Final Test mới',exact=True).click()
                fixtures=[]
                for i in (9,19):
                    path=Path(folder)/f'fixture{i}.png'
                    rng=np.random.default_rng(i)
                    cv2.imwrite(str(path),cv2.resize(rng.integers(0,256,(16,16,3),dtype=np.uint8),(160,160)))
                    fixtures.append(str(path))
                page.locator('input[type=file]').set_input_files(fixtures)
                page.get_by_text('fixture19.png',exact=True).first.wait_for()
                page.get_by_role('button',name='Nhập ảnh vào hàng đợi').click()
                expect(page.get_by_test_id('stMetricValue')).to_have_text(['0','42','0','0'],timeout=60000)
                page.get_by_role('button',name='Nhập ảnh vào hàng đợi').click()
                page.get_by_text('fixture9.png: Ảnh trùng SHA-256',exact=True).wait_for(timeout=60000)
                assert not store.all_latest()
                page.get_by_text('OCR Pilot Test',exact=True).click()
                page.get_by_text('từng xuất hiện trong diagnostics',exact=False).wait_for()
                expect(page.get_by_test_id('stMetricValue')).to_have_text(['3','0','3','24'])
                expect(page.get_by_text('Tôi xác nhận box biển',exact=True)).to_have_count(1)
                page.get_by_text('Crop box thủ công',exact=True).wait_for()
                page.get_by_text('Tôi xác nhận box biển',exact=True).click()
                page.get_by_text('Tôi xác nhận text',exact=True).click()
                for i in range(8):
                    checkbox=page.get_by_role('checkbox',name='Đã kiểm tra glyph đủ nét').nth(i)
                    page.get_by_text('Đã kiểm tra glyph đủ nét',exact=True).nth(i).click()
                    expect(checkbox).to_be_checked()
                    page.wait_for_timeout(550)
                page.get_by_text('Tôi xác nhận toàn bộ box, glyph và thứ tự',exact=True).click()
                page.get_by_role('button',name='Lưu Glyph reviewed').click(timeout=10000)
                for _ in range(40):
                    if store.all_latest():break
                    page.wait_for_timeout(250)
                assert len(store.all_latest())==1 and next(iter(store.all_latest().values()))['cohort']=='pilot'
                expect(page.get_by_test_id('stMetricValue')).to_have_text(['3','0','3','24'])
                assert not page.get_by_test_id('stException').count()
                page.screenshot(path=out/'pilot_review.png');b.close()
            (out/'results.json').write_text(json.dumps({'status':'PASS','plate_canvas':'PASS','manual_canvas':'PASS','autosave_reload':'PASS',
                'unverified_final_blocked':True,'pilot_review_save':'PASS','multi_upload':'PASS','duplicate_rejected':'PASS','production_DB_untouched':True}),encoding='utf-8')
        finally:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait(timeout=10)


if __name__=='__main__':main()