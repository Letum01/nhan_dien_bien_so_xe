"""Chromium acceptance using an isolated DB and a procedural image only."""
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.request import urlopen

import cv2
import numpy as np
from playwright.sync_api import expect, sync_playwright

from learning_center import Feedback


def main():
    root=Path(__file__).resolve().parent
    with TemporaryDirectory(prefix='learning_fixture_') as tmp:
        tmp=Path(tmp); db=tmp/'feedback.sqlite3'; path=tmp/'fixture.png'
        image=np.zeros((180,420,3),np.uint8); image[45:132,42:385]=225
        cv2.putText(image,'12A34567',(52,108),cv2.FONT_HERSHEY_SIMPLEX,1.7,(15,15,15),3)
        assert cv2.imwrite(str(path),image)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        with (tmp/'server.log').open('wb') as log:
            process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(root/'app_learning_center.py'),
                '--server.address=127.0.0.1',f'--server.port={port}','--server.headless=true'],cwd=root,
                env={**os.environ,'ALPR_LEARNING_DB':str(db)},stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=monotonic()+25
                while True:
                    try:
                        with urlopen(url+'/_stcore/health',timeout=1) as r: assert r.read()==b'ok'
                        break
                    except OSError:
                        if monotonic()>deadline or process.poll() is not None:raise
                        sleep(.3)
                with sync_playwright() as p:
                    browser=p.chromium.launch(); page=browser.new_page(accept_downloads=True); page.goto(url)
                    expect(page.get_by_role('heading',name='ALPR Learning Center')).to_be_visible(timeout=45000)
                    def upload(allow):
                        page.reload()
                        expect(page.get_by_role('heading',name='ALPR Learning Center')).to_be_visible(timeout=45000)
                        page.locator('input[type=file]').set_input_files(path)
                        expect(page.get_by_text('fixture.png',exact=True)).to_be_visible(timeout=15000)
                        page.get_by_role('textbox',name='Nguồn ảnh (không nhập thông tin cá nhân)').fill('Procedural browser fixture')
                        page.get_by_role('textbox',name='Source group đã biết').fill('browser-fixture')
                        if allow:page.get_by_text('Tôi đồng ý lưu ảnh này để nghiên cứu local',exact=True).click()
                        page.get_by_role('button',name='Lưu ảnh nghiên cứu').click()
                    upload(False);expect(page.get_by_text('Chọn ảnh và đồng ý lưu trước khi tiếp tục.')).to_be_visible()
                    assert not Feedback(db).samples()
                    upload(True)
                    try: expect(page.get_by_text('Đã lưu ảnh nghiên cứu.')).to_be_visible(timeout=45000)
                    except AssertionError as exc: raise AssertionError(page.locator('main').inner_text()[:3000]) from exc
                    rows=Feedback(db).samples();assert len(rows)==1 and rows[0]['raw'].get('model_hashes')
                    upload(True)
                    try:expect(page.get_by_text('Ảnh trùng SHA-256.')).to_be_visible(timeout=15000)
                    except AssertionError as exc:raise AssertionError(page.locator('main').inner_text()[:3000]) from exc
                    for text,rev in [('12A34567',2),('12A34568',3)]:
                        page.get_by_role('textbox',name='Text thật',exact=False).fill(text)
                        page.get_by_text('Tôi xác nhận đã đọc text từ ảnh',exact=True).click()
                        page.get_by_role('button',name='Lưu xác nhận').click()
                        expect(page.get_by_text(f'Revision {rev} ·',exact=False)).to_be_visible()
                    page.reload();expect(page.get_by_role('textbox',name='Text thật',exact=False)).to_have_value('12A34568',timeout=45000)
                    page.get_by_text('Cho phép xuất metadata nghiên cứu',exact=True).click()
                    with page.expect_download() as download:page.get_by_role('button',name='Tải JSON',exact=False).click()
                    import json
                    result=json.loads(download.value.path().read_text(encoding='utf-8'))
                    assert not result['training_allowed'] and len(result['sample']['revisions'])==3
                    page.get_by_text('Tôi muốn xóa ảnh và toàn bộ lịch sử mẫu này',exact=True).click()
                    page.get_by_role('button',name='Xóa mẫu').click();expect(page.get_by_text('Chưa có mẫu nghiên cứu.')).to_be_visible()
                    assert not Feedback(db).samples() and not page.get_by_test_id('stException').count()
                    print('Chromium PASS: opt-in, CUDA prediction, duplicate, revisions, reload, export, delete; glyph boxes:',len(rows[0]['raw']['glyphs']))
            finally:
                if process.poll() is None:
                    if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,check=False)
                    else:process.terminate()
                process.wait(timeout=15)


if __name__=='__main__':main()