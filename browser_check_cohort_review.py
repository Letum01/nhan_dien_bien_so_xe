"""Exercise the cohort UI on the real read-only queue with a temporary response DB."""
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

from cohort_review import ROOT, Reviews


def main():
    with TemporaryDirectory() as folder:
        db=Path(folder)/'fixture.sqlite3'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        url=f'http://127.0.0.1:{port}'; log=Path(folder)/'server.log'
        with log.open('wb') as stream:
            proc=subprocess.Popen([sys.executable,'-m','streamlit','run',str(ROOT/'app_learning_center.py'),
                '--server.address=127.0.0.1',f'--server.port={port}','--server.headless=true'],cwd=ROOT,
                env={**os.environ,'ALPR_COHORT_REVIEW_DB':str(db)},stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline=monotonic()+25
                while True:
                    try:
                        with urlopen(url+'/_stcore/health',timeout=1) as r: assert r.read()==b'ok'
                        break
                    except OSError:
                        if monotonic()>deadline: raise
                        sleep(.25)
                with sync_playwright() as p:
                    browser=p.chromium.launch(); page=browser.new_page(accept_downloads=True)
                    page.goto(url+'/?mode=cohort')
                    expect(page.get_by_role('heading',name=re.compile('^Nhãn nguồn: [0-9A-Z]$'))).to_be_visible(timeout=30000)
                    page.get_by_text('Mã mẫu gốc',exact=True).click()
                    page.get_by_role('textbox',name='Bí danh người rà soát').fill('temporary browser check')
                    page.get_by_role('button',name='Đúng',exact=True).click()
                    expect(page.get_by_text('Glyph: 1/412',exact=False)).to_be_visible()
                    expect(page.get_by_role('spinbutton')).to_have_value('2')
                    page.get_by_text('Độ đầy đủ box (122)',exact=True).click()
                    expect(page.get_by_role('heading',name=re.compile('box trong annotation nguồn'))).to_be_visible()
                    page.get_by_role('button',name='Thiếu box',exact=True).click()
                    expect(page.get_by_text('Độ đầy đủ box: 1/122',exact=False)).to_be_visible()
                    page.reload()
                    expect(page.get_by_text('Glyph: 1/412',exact=False)).to_be_visible(timeout=20000)
                    page.get_by_text('Mã mẫu gốc',exact=True).click()
                    expect(page.get_by_role('spinbutton')).to_have_value('2')
                    page.get_by_text('Thống kê và xuất kết quả',exact=True).click()
                    expect(page.get_by_text('Sai nhãn/crop trên mẫu đã kết luận: 0/1',exact=False)).to_be_visible()
                    with page.expect_download(timeout=12000) as event:
                        page.get_by_role('button',name='Tải CSV tiến độ',exact=True).click()
                    with event.value.path().open(encoding='utf-8-sig') as f: assert '1,correct,' in f.read()
                    page.get_by_text('Review nhiều glyph cùng ảnh',exact=True).click()
                    expect(page.get_by_role('heading',name='Các glyph trong cùng ảnh')).to_be_visible()
                    page.get_by_role('textbox',name='Bí danh người rà soát').fill('temporary group check')
                    # Select two distinct glyphs separately; remaining widgets stay unset.
                    labels=page.get_by_text(re.compile(r'^Quyết định #\d+$'))
                    expect(labels.first).to_be_visible()
                    ids=[int(t.split('#')[1]) for t in labels.all_text_contents()[:2]]
                    assert len(ids)==2
                    selectors=page.get_by_test_id('stSelectbox').filter(has_text=re.compile('Quyết định #'))
                    for i,choice in enumerate(('Sai','Không rõ')):
                        selectors.nth(i).get_by_role('combobox').click()
                        page.get_by_role('option',name=choice,exact=True).click()
                    page.get_by_role('button',name='Lưu các quyết định đã chọn').click()
                    expect(page.get_by_text('Glyph: 3/412',exact=False)).to_be_visible()
                    browser.close()
                store=Reviews(db)
                assert store.latest('glyph')[1]['decision']=='correct'
                assert {i:store.latest('glyph')[i]['decision'] for i in ids}==dict(zip(ids,('incorrect','unclear')))
                assert len(store.latest('glyph'))==3
                assert store.latest('detector')[1]['decision']=='incomplete'
                try: store.save('glyph',1,0,'incorrect','unknown','','','test')
                except ValueError: pass
                else: raise AssertionError('Stale revision accepted')
                print('Chromium: open, glyph and detector save, auto-advance, reload, CSV, stale revision PASS')
            except Exception:
                print(log.read_text(encoding='utf-8',errors='replace')); raise
            finally:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                proc.wait()


if __name__=='__main__': main()