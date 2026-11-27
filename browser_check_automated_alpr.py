"""Read-only live smoke for the two isolated workflow apps."""
import json
from playwright.sync_api import expect, sync_playwright
from glyph_review import ROOT


def main():
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page()
        page.goto('http://localhost:8510')
        expect(page.get_by_role('heading',name='Final Test — nhập text nhanh')).to_be_visible(timeout=90000)
        expect(page.get_by_text('GLYPH_REVIEWED hiện hữu, chỉ đọc')).to_be_visible(timeout=90000)
        page.get_by_role('combobox').click()
        page.get_by_role('option',name='2. IMG_2237.jpg',exact=True).click()
        expect(page.get_by_role('button',name='Lưu TEXT_REVIEWED')).to_be_disabled(timeout=90000)
        assert not page.get_by_text('Plate text dự đoán',exact=True).count()
        page.screenshot(path=ROOT/'reports/automated_blind_browser.png')
        page.goto('http://localhost:8509')
        page.get_by_text('Auto Labeling — gắn nhãn tự động',exact=True).click(timeout=90000)
        page.get_by_role('button',name='Mở batch đề xuất đã lưu').click(timeout=90000)
        expect(page.get_by_label('Sửa chuỗi bằng bàn phím')).to_be_visible(timeout=90000)
        assert not page.get_by_test_id('stException').count()
        page.screenshot(path=ROOT/'reports/automated_dev_browser.png')
        browser.close()
    (ROOT/'reports/automated_browser.json').write_text(json.dumps({'blind_gate':'PASS','development_batch_display':'PASS','reviews_created':False}),encoding='utf-8')


if __name__=='__main__':main()