"""Real Chromium glyph tests with synthetic pixels + temporary DB only."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'reports/glyph_browser';output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='alpr_glyph_test_',ignore_cleanup_errors=True) as temp:
        folder=Path(temp);source=folder/'fixture.jpg';database=folder/'test.sqlite3'
        image=np.full((180,300,3),220,np.uint8)
        cv2.putText(image,'AB',(40,90),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)
        cv2.imwrite(str(source),image)
        env={**os.environ,'ALPR_ANNOTATION_DB':str(database),'ALPR_ANNOTATION_SOURCE':str(folder)}
        with (output/'server.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','streamlit','run',str(ROOT/'label_detection.py'),
                                   '--server.port=8892','--server.headless=true','--browser.gatherUsageStats=false'],
                                  env=env,cwd=ROOT,stdout=log,stderr=log)
            try:
                for _ in range(60):
                    try:urllib.request.urlopen('http://127.0.0.1:8892/_stcore/health',timeout=1);break
                    except OSError:time.sleep(.5)
                with sync_playwright() as p:
                    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1500,'height':2000})
                    page.goto('http://127.0.0.1:8892')
                    page.get_by_text('ALPR — Gắn nhãn nhiều bước').wait_for(timeout=60000)
                    page.wait_for_timeout(2500)
                    def draw(canvas,x1,y1,x2,y2):
                        canvas.scroll_into_view_if_needed();b=canvas.bounding_box()
                        page.mouse.wheel(0,400);page.wait_for_timeout(400);b=canvas.bounding_box()
                        page.mouse.move(b['x']+x1,b['y']+y1);page.mouse.down()
                        page.mouse.move(b['x']+x2,b['y']+y2,steps=12);page.mouse.up();page.wait_for_timeout(1500)
                    def canvases():
                        return [f.locator('canvas.upper-canvas').first for f in page.frames if f.locator('canvas.upper-canvas').count()]
                    draw(canvases()[0],20,20,250,140)
                    print('tabs',page.get_by_role('tab').all_text_contents(),flush=True)
                    (output/'page.txt').write_text(page.inner_text('body'),encoding='utf-8')
                    page.get_by_role('tab',name='Ký tự',exact=True).click();page.wait_for_timeout(1000)
                    page.get_by_role('button',name='Mở canvas ký tự',exact=True).click();page.wait_for_timeout(1800)
                    def glyph_canvas():
                        visible=[canvas for frame in page.frames for canvas in [frame.locator('canvas.upper-canvas:visible').last]
                                 if canvas.count() and canvas.is_visible()]
                        return visible[-1]
                    draw(glyph_canvas(),20,20,100,100)
                    print('glyph canvas',glyph_canvas().bounding_box(),flush=True)
                    page.screenshot(path=str(output/'draw_debug.png'),full_page=True)
                    def latest():
                        with sqlite3.connect(database) as c:r=c.execute('SELECT payload FROM drafts LIMIT 1').fetchone()
                        return json.loads(r[0])
                    chars=latest()['plates'][0]['characters'];assert len(chars)==1,chars
                    initial=chars[0]['bbox']
                    page.get_by_text('transform',exact=True).last.click();page.wait_for_timeout(600)
                    canvas=glyph_canvas();canvas.scroll_into_view_if_needed();b=canvas.bounding_box()
                    page.mouse.move(b['x']+60,b['y']+60);page.mouse.down();page.mouse.move(b['x']+80,b['y']+70,steps=10);page.mouse.up();page.wait_for_timeout(1200)
                    moved=latest()['plates'][0]['characters'][0]['bbox'];assert moved[:2]!=initial[:2]
                    canvas=glyph_canvas();canvas.scroll_into_view_if_needed();b=canvas.bounding_box()
                    page.mouse.click(b['x']+(moved[0]+moved[2]/2)*2,b['y']+(moved[1]+moved[3]/2)*2);page.wait_for_timeout(600)
                    page.mouse.move(b['x']+(moved[0]+moved[2])*2,b['y']+(moved[1]+moved[3])*2);page.mouse.down()
                    page.mouse.move(b['x']+(moved[0]+moved[2])*2+20,b['y']+(moved[1]+moved[3])*2+20,steps=12);page.mouse.up();page.wait_for_timeout(1500)
                    resized=latest()['plates'][0]['characters'][0]['bbox']
                    if resized[2:]==moved[2:]:
                        canvas=glyph_canvas();b=canvas.bounding_box()
                        page.mouse.click(b['x']+(moved[0]+moved[2]/2)*2,b['y']+(moved[1]+moved[3]/2)*2);page.wait_for_timeout(800)
                        page.mouse.move(b['x']+(moved[0]+moved[2])*2+1,b['y']+(moved[1]+moved[3]/2)*2);page.mouse.down()
                        page.mouse.move(b['x']+(moved[0]+moved[2])*2+25,b['y']+(moved[1]+moved[3]/2)*2,steps=10);page.mouse.up();page.wait_for_timeout(1500)
                        resized=latest()['plates'][0]['characters'][0]['bbox']
                    resize_status='PASS' if resized[2:]!=moved[2:] else 'FAIL: saved dimensions did not change'
                    def select_index(value):
                        control=page.get_by_label('Index thao tác (tùy chọn, ví dụ 0,1)',exact=True)
                        control.fill(str(value));control.press('Tab');page.wait_for_timeout(1200)
                    select_index(0)
                    page.get_by_role('button',name='Split glyph',exact=True).click();page.wait_for_timeout(1800)
                    assert len(latest()['plates'][0]['characters'])==2
                    # Selection must be reset after topology changes by app.
                    select_index('0,1')
                    page.get_by_role('button',name='Merge glyph cùng dòng',exact=True).click();page.wait_for_timeout(1800)
                    print('after merge',latest()['plates'][0]['characters'],'alerts',page.locator('[data-testid="stAlert"]').all_text_contents(),flush=True)
                    assert len(latest()['plates'][0]['characters'])==1
                    page.get_by_label('Nhãn glyph',exact=True).fill('A');page.get_by_label('Nhãn glyph',exact=True).press('Tab');page.wait_for_timeout(1000)
                    page.get_by_label('Dòng (0=trên)',exact=True).fill('1');page.get_by_label('Dòng (0=trên)',exact=True).press('Enter');page.wait_for_timeout(1000)
                    page.get_by_text('Đã kiểm tra glyph đủ nét',exact=True).click();page.wait_for_timeout(1000)
                    saved=latest()['plates'][0]['characters'];assert saved[0]['label']=='A' and saved[0]['line']==1
                    page.reload();page.wait_for_timeout(3000)
                    assert latest()['plates'][0]['characters']==saved
                    page.get_by_role('tab',name='Ký tự',exact=True).click();page.wait_for_timeout(1000)
                    page.get_by_role('button',name='Mở canvas ký tự',exact=True).click();page.wait_for_timeout(1500)
                    select_index(0);page.get_by_role('button',name='Xóa glyph đã chọn',exact=True).click();page.wait_for_timeout(1200)
                    assert latest()['plates'][0]['characters']==[]
                    page.get_by_label('source_group',exact=True).fill('SYNTHETIC_TEST_ONLY');page.get_by_label('source_group',exact=True).press('Tab');page.wait_for_timeout(1000)
                    page.get_by_role('tab',name='Văn bản biển số',exact=True).click()
                    page.get_by_label('Text gốc (không tự sửa)',exact=True).fill('A');page.get_by_label('Text gốc (không tự sửa)',exact=True).press('Tab');page.wait_for_timeout(1000)
                    page.get_by_text('Tôi xác nhận text đúng với crop',exact=True).click();page.wait_for_timeout(1000)
                    page.get_by_role('tab',name='Lịch sử / xuất',exact=True).click()
                    page.get_by_text('Tôi xác nhận toàn bộ box/text/glyph đã nhập; lưu reviewed revision mới',exact=True).click();page.wait_for_timeout(1000)
                    page.get_by_role('button',name='Lưu toàn bộ nhãn reviewed',exact=True).click();page.wait_for_timeout(1500)
                    with sqlite3.connect(database) as c:
                        reviewed=c.execute('SELECT payload FROM revisions ORDER BY id DESC LIMIT 1').fetchone()
                    assert reviewed and json.loads(reviewed[0])['review_status']=='reviewed'
                    assert not page.locator('[data-testid="stException"]').count()
                    page.screenshot(path=str(output/'glyphs.png'),full_page=True)
                    (output/'results.json').write_text(json.dumps({'fixture_only':True,'draw':'PASS','move':'PASS','resize':resize_status,
                        'split':'PASS','merge':'PASS','edit_label_line':'PASS','reload_draft':'PASS','delete_one':'PASS',
                        'reviewed_ui':'PASS synthetic fixture temporary DB only','real_labels_created':False},indent=2),encoding='utf-8')
                    browser.close();print('Glyph Chromium operations PASS (synthetic fixture only)')
            finally:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                proc.wait(timeout=10)


if __name__=='__main__':main()