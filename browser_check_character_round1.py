"""Chromium upload smoke against the isolated Round 1 UI."""
import json
import re
from pathlib import Path
from time import perf_counter

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'reports/character_round1_browser'; out.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(); page=browser.new_page(viewport={'width':1360,'height':1000})
        page.goto('http://localhost:8507'); page.get_by_role('heading',name='YOLO Character — thử nghiệm').wait_for(timeout=60000)
        errors=[]; page.on('pageerror',lambda e:errors.append(str(e)))
        page.locator('input[type=file]').set_input_files(ROOT/'dataset/license_plate_dataset/0/images/14.jpg')
        def select(label,value):
            page.get_by_test_id('stSelectbox').filter(has=page.get_by_text(label,exact=True)).get_by_role('combobox').click()
            page.get_by_role('option',name=value,exact=True).click()
        select('Tách ký tự','YOLO Character')
        records=[]
        for backend in ('SVM Mix','Random Forest Mix','CNN Mix','Majority Ensemble (Mix)'):
            current=page.get_by_test_id('stSelectbox').nth(1).inner_text()
            if backend not in current:
                select('Nhận dạng',backend)
                page.get_by_role('heading',name='Plate text dự đoán').wait_for(state='hidden')
            start=perf_counter(); page.get_by_role('button',name='Chạy thử nghiệm').click()
            page.get_by_role('heading',name='Plate text dự đoán').wait_for(timeout=120000)
            assert not page.get_by_test_id('stException').count()
            text=page.locator('pre code').inner_text().strip(); assert text and text!='(trống)'
            timing=page.get_by_text(re.compile(r'^Streamlit enqueue')).inner_text()
            worker=page.get_by_text(re.compile(r'^Worker PID')).inner_text().strip()
            records.append({'backend':backend,'output':text,'browser_seconds_to_metrics':perf_counter()-start,
                            'server_render_caption':timing,'worker':worker})
        assert len({r['worker'].split(' · ')[0] for r in records})==1
        assert all(r['worker'].endswith('warm') for r in records[1:]),records
        page.get_by_text('Glyph previews',exact=True).click()
        page.screenshot(path=out/'upload.png',full_page=True)
        assert not errors,errors
        (out/'results.json').write_text(json.dumps({'status':'PASS','image':'14.jpg','backends':records},indent=2),encoding='utf-8')
        browser.close()


if __name__=='__main__':main()