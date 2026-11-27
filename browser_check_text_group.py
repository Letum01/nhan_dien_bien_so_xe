"""Replay the actual failed draft using a temporary copy, never saving live labels."""
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright
from quick_ocr_test import ROOT, TEXT_STORE, ReadOnlyStore, audit_candidates, entries, key
from annotation_store import AnnotationStore


def main():
    items=entries(audit_candidates(write=False))
    entry=next(e for e in items if e['filename']=='bien-so-xe-cac-tinh-4-1.jpg')
    ident=key(entry); position=items.index(entry)+1
    with TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path=Path(folder)/'text.sqlite3'
        with sqlite3.connect(TEXT_STORE.as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(path) as dst:src.backup(dst)
        store=AnnotationStore(path)
        draft=store.draft(ident)[0]; assert draft['source_group']=='final-may-002'
        env={**os.environ,'ALPR_TEXT_DB':str(path)}
        url='http://127.0.0.1:8899'
        process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(ROOT/'app_quick_ocr_test.py'),
                                  '--server.address=127.0.0.1','--server.port=8899','--server.headless=true'],cwd=ROOT,env=env)
        try:
            for _ in range(60):
                try:
                    with urlopen(url+'/_stcore/health',timeout=1):break
                except OSError:time.sleep(.5)
            with sync_playwright() as p:
                browser=p.chromium.launch();page=browser.new_page();page.goto(url)
                def select():
                    page.get_by_test_id('stSelectbox').filter(has=page.get_by_text('Ảnh trong hàng đợi',exact=True)).get_by_role('combobox').click()
                    page.get_by_role('option',name=f'{position}. {entry["filename"]}',exact=True).click()
                select()
                expect(page.get_by_label('Text biển nhập tay')).to_have_value(draft['plate_text'],timeout=60000)
                page.get_by_text('Lưu nguồn này để dùng lại cho ảnh thực sự cùng nguồn',exact=True).click()
                page.get_by_text('Tôi xác nhận text đọc từ ảnh và nguồn độc lập, chưa dùng để phát triển',exact=True).click()
                page.get_by_role('button',name='Lưu TEXT_REVIEWED').click()
                expect(page.get_by_text('Đã lưu TEXT_REVIEWED: '+entry['filename'],exact=True)).to_be_visible(timeout=30000)
                expect(page.get_by_test_id('stSelectbox').first).to_contain_text(items[position]['filename'],timeout=60000)
                assert ReadOnlyStore(path).all_latest()[ident]['plate_text']==draft['plate_text']
                page.reload();select()
                expect(page.get_by_label('Text biển nhập tay')).to_have_value(draft['plate_text'],timeout=60000)
                page.get_by_test_id('stSelectbox').filter(has=page.get_by_text('Nguồn ảnh',exact=True)).get_by_role('combobox').click()
                expect(page.get_by_role('option',name='final-may-002',exact=True)).to_be_visible()
                assert not page.get_by_test_id('stException').count()
                browser.close()
            print('PASS original draft replay, same-group save, auto-next, reload and saved source; temporary DB only')
        finally:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True);process.wait(timeout=10)


if __name__=='__main__':main()