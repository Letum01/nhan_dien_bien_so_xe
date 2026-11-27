"""Smoke-test Deep mode with development image only; no pixel saving."""
from pathlib import Path
from playwright.sync_api import expect,sync_playwright

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    with sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page();page.goto('http://localhost:8512')
        page.get_by_text('Deep Analysis Experimental',exact=True).click()
        page.locator('input[type=file]').set_input_files(ROOT/'dataset/license_plate_dataset/0/images/14.jpg')
        page.get_by_role('button',name='Phân tích',exact=True).click()
        expect(page.get_by_text('Crop và thứ tự box glyph',exact=True)).to_be_visible(timeout=90000)
        expect(page.get_by_text('Raw CNN:',exact=False)).to_be_visible()
        expect(page.get_by_text('Top-k là điểm nội bộ',exact=False)).to_be_visible()
        assert not page.get_by_test_id('stException').count()
        page.screenshot(path=ROOT/'reports/accuracy_recovery/deep_browser.png');browser.close()
    print('Deep Analysis Chromium PASS')