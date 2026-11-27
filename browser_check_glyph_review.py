"""Focused glyph UI browser smoke; temporary OCR DB, no human-review writes."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    train_queue=os.environ.get('ALPR_OCR_QUEUE')=='train'
    hard_queue=os.environ.get('ALPR_OCR_QUEUE')=='hard'
    output=ROOT/'reports'/('glyph_hard_browser' if hard_queue else 'glyph_train_browser' if train_queue else 'glyph_review_browser');output.mkdir(exist_ok=True)
    detection=ROOT/'data/annotations.sqlite3'
    with sqlite3.connect(f'file:{detection.as_posix()}?mode=ro',uri=True) as connection:
        before=connection.execute('SELECT * FROM revisions ORDER BY id').fetchall()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        db=Path(folder)/'ocr.db';env={**os.environ,'ALPR_GLYPH_DB':str(db)}
        with (output/'server.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','streamlit','run','label_characters.py','--server.port=8896',
                                   '--server.headless=true','--browser.gatherUsageStats=false'],cwd=ROOT,env=env,stdout=log,stderr=log)
            try:
                for _ in range(60):
                    try:urllib.request.urlopen('http://127.0.0.1:8896/_stcore/health',timeout=1);break
                    except OSError:time.sleep(.5)
                with sync_playwright() as p:
                    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1500,'height':1800})
                    page.goto('http://127.0.0.1:8896');page.get_by_text('Gắn nhãn OCR / Glyph',exact=True).wait_for(timeout=60000)
                    if train_queue:
                        page.get_by_role('heading',name='OCR Train — ưu tiên',exact=True).wait_for()
                        assert 'train' in page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text('Biển cần gắn glyph',exact=True)).inner_text()
                    if hard_queue:
                        page.get_by_role('heading',name='OCR Hard Cases — ưu tiên',exact=True).wait_for()
                        assert 'train' in page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text('Biển cần gắn glyph',exact=True)).inner_text()
                    page.get_by_role('button',name='Tạo glyph proposal chưa nhãn',exact=True).click();page.wait_for_timeout(2500)
                    def payload():
                        with sqlite3.connect(db) as c:return json.loads(c.execute('SELECT payload FROM drafts LIMIT 1').fetchone()[0])
                    first=payload()
                    if hard_queue and len(first['characters'])<=1:
                        # Hard crops legitimately fail segmentation; preserve this
                        # case, then select another actual queue image for delete.
                        control=page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text('Biển cần gắn glyph',exact=True)).get_by_role('combobox')
                        for candidate_index in range(1,10):
                            control.click();page.get_by_role('option').nth(candidate_index).click();page.wait_for_timeout(1200)
                            page.get_by_role('button',name='Tạo glyph proposal chưa nhãn',exact=True).click();page.wait_for_timeout(1800)
                            # Fetch current plate draft, not the first previously visited draft.
                            with sqlite3.connect(db) as c:
                                drafts=[json.loads(r[0]) for r in c.execute('SELECT payload FROM drafts ORDER BY rowid DESC')]
                            first=drafts[0]
                            if len(first['characters'])>1:break
                        active_id=first['image_id']
                        def payload():
                            with sqlite3.connect(db) as c:
                                return json.loads(c.execute('SELECT payload FROM drafts WHERE image_id=?',(active_id,)).fetchone()[0])
                    assert len(first['characters'])>1
                    count=len(first['characters'])
                    field=page.get_by_label('Index thao tác (tùy chọn, ví dụ 0,1)',exact=True)
                    field.fill('0');field.press('Tab');page.wait_for_timeout(1000)
                    page.get_by_role('button',name='Xóa glyph đã chọn',exact=True).click();page.wait_for_timeout(1500)
                    assert len(payload()['characters'])==count-1
                    page.reload();page.wait_for_timeout(3000)
                    assert len(payload()['characters'])==count-1
                    assert not page.locator('[data-testid="stException"]').count()
                    assert page.get_by_role('button',name='Lưu Glyph reviewed',exact=True).is_disabled()
                    page.screenshot(path=str(output/'glyph_review.png'),full_page=True)
                    with sqlite3.connect(db) as c:assert c.execute('SELECT count(*) FROM revisions').fetchone()[0]==0
                    browser.close()
            finally:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait(timeout=10)
    with sqlite3.connect(f'file:{detection.as_posix()}?mode=ro',uri=True) as c:assert c.execute('SELECT * FROM revisions ORDER BY id').fetchall()==before
    (output/'results.json').write_text(json.dumps({'render':'PASS','proposal_creation':'PASS','delete_noise_by_index':'PASS',
        'draft_reload':'PASS','incomplete_review_blocked':'PASS','detection_revisions_unchanged':'PASS',
        'ocr_database':'temporary','reviewed_labels_created':False,
        'canvas_mouse_split_merge':'NOT_TESTED_in_this_focused_run; previous fixture browser tests separate'},indent=2),encoding='utf-8')
    print('Focused glyph Chromium PASS; no reviewed labels created; detection revisions unchanged')


if __name__=='__main__':main()