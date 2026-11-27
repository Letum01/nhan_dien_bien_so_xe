"""Check the live test queue without creating review or editing glyphs."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent

with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page()
    page.goto('http://localhost:8508')
    page.get_by_role('heading',name='OCR Test — đánh giá độc lập').wait_for(timeout=60000)
    page.get_by_role('button',name='Lưu Glyph reviewed').wait_for(timeout=60000)
    assert not page.get_by_role('button',name='Tạo glyph proposal chưa nhãn').count()
    assert not page.get_by_test_id('stException').count()
    page.screenshot(path=ROOT/'reports/ocr_test_queue.png')
    browser.close()
(ROOT/'reports/ocr_test_browser.json').write_text(json.dumps({'status':'PASS','proposal_hidden':True,'reviews_created':False}),encoding='utf-8')