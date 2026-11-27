"""Explicit browser acceptance check on temporary annotation DB, no real labels."""
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
    with tempfile.TemporaryDirectory(prefix='alpr_browser_',ignore_cleanup_errors=True) as temporary:
        folder=Path(temporary);database=folder/'annotations.db'
        env={**os.environ,'ALPR_ANNOTATION_DB':str(database)}
        with (folder/'server.log').open('w',encoding='utf-8') as logs:
            process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(ROOT/'label_detection.py'),
                                      '--server.headless=true','--server.port=8891','--browser.gatherUsageStats=false'],
                                     cwd=ROOT,env=env,stdout=logs,stderr=logs)
            try:
                for _ in range(60):
                    try:
                        urllib.request.urlopen('http://127.0.0.1:8891/_stcore/health',timeout=1);break
                    except OSError:time.sleep(.5)
                with sync_playwright() as p:
                    browser=p.chromium.launch(headless=True)
                    page=browser.new_page(viewport={'width':1450,'height':1100})
                    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                    page.goto('http://127.0.0.1:8891');page.get_by_text('ALPR — Gắn nhãn nhiều bước').wait_for(timeout=60000)
                    page.wait_for_timeout(3000)
                    frames=[f for f in page.frames if f.locator('canvas.upper-canvas').count()]
                    if not frames:raise AssertionError('Canvas frame not mounted')
                    canvas=frames[0].locator('canvas.upper-canvas').first
                    canvas.scroll_into_view_if_needed()
                    box=canvas.bounding_box()
                    page.mouse.move(box['x']+40,box['y']+45);page.mouse.down()
                    page.mouse.move(box['x']+170,box['y']+130,steps=12);page.mouse.up()
                    page.wait_for_timeout(3500)
                    print('canvas box',box,'exceptions',page.locator('[data-testid="stException"]').all_text_contents(),flush=True)
                    def draft_boxes():
                        with sqlite3.connect(database) as connection:
                            rows=connection.execute('SELECT payload FROM drafts').fetchall()
                        return [json.loads(row[0]) for row in rows]
                    drawn=draft_boxes()
                    assert any(row.get('boxes') for row in drawn),'Mouse drawing not autosaved'
                    first=next(row for row in drawn if row.get('boxes'))
                    original_box=first['boxes'][0]
                    page.get_by_text('transform',exact=True).first.click();page.wait_for_timeout(1000)
                    frame=next(f for f in page.frames if f.locator('canvas.upper-canvas').count())
                    target=frame.locator('canvas.upper-canvas').first;target.scroll_into_view_if_needed()
                    bounds=target.bounding_box()
                    # Drag rectangle interior by a visible offset.
                    page.mouse.move(bounds['x']+90,bounds['y']+85);page.mouse.down()
                    page.mouse.move(bounds['x']+115,bounds['y']+100,steps=10);page.mouse.up();page.wait_for_timeout(2000)
                    moved=next(row for row in draft_boxes() if row['image_id']==first['image_id'])['boxes'][0]
                    assert moved[:2]!=original_box[:2],('move did not persist',original_box,moved)
                    # Select and drag bottom-right Fabric resize handle.
                    target.scroll_into_view_if_needed();bounds=target.bounding_box()
                    page.mouse.click(bounds['x']+moved[0]+moved[2]/2,bounds['y']+moved[1]+moved[3]/2)
                    page.wait_for_timeout(500)
                    page.mouse.move(bounds['x']+moved[0]+moved[2],bounds['y']+moved[1]+moved[3]);page.mouse.down()
                    page.mouse.move(bounds['x']+moved[0]+moved[2]+25,bounds['y']+moved[1]+moved[3]+20,steps=10)
                    page.mouse.up();page.wait_for_timeout(2000)
                    resized=next(row for row in draft_boxes() if row['image_id']==first['image_id'])['boxes'][0]
                    resize_status='PASS' if resized[2:]!=moved[2:] else 'FAIL: mouse handle did not change saved dimensions'
                    page.get_by_role('button',name='Ảnh tiếp theo →',exact=True).click();page.wait_for_timeout(1500)
                    page.get_by_role('button',name='← Ảnh trước',exact=True).click();page.wait_for_timeout(1500)
                    assert next(row for row in draft_boxes() if row['image_id']==first['image_id'])['boxes']==[resized]
                    # Reload opens saved last image and draft, without approval.
                    page.reload();page.wait_for_timeout(3500)
                    frames=[f for f in page.frames if f.locator('canvas.upper-canvas').count()]
                    assert frames
                    assert not page.locator('[data-testid="stException"]').count()
                    page.get_by_role('button',name='Xóa tất cả box / từ chối proposal',exact=True).click();page.wait_for_timeout(1500)
                    assert next(row for row in draft_boxes() if row['image_id']==first['image_id'])['boxes']==[]
                    output=ROOT/'reports'/'annotation_browser'
                    output.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(output/'canvas_autosave.png'),full_page=True)
                    (output/'results.json').write_text(json.dumps({'draw_and_autosave':'PASS','previous_next':'PASS',
                        'reload_draft':'PASS','reviewed_labels_written':False,'browser_errors':errors,
                        'move':'PASS','resize':resize_status,'delete_all':'PASS',
                        'glyph_canvas_split_merge':'NOT TESTED'},indent=2),encoding='utf-8')
                    assert not errors,errors
                    browser.close()
                    print('Browser draw/autosave/navigation/reload PASS; no human-reviewed labels created')
            finally:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()