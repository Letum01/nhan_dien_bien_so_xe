"""Acceptance smoke for a running demo, with history opt-in left off."""
from pathlib import Path
import json
import tempfile

import cv2
import numpy as np
from playwright.sync_api import expect,sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    real=ROOT/'dataset/license_plate_dataset/0/images/14.jpg'
    with tempfile.TemporaryDirectory() as tmp,sync_playwright() as p:
        bad=Path(tmp)/'broken.png';bad.write_bytes(b'not an image')
        blank=Path(tmp)/'blank.png';assert cv2.imwrite(str(blank),np.zeros((1300,2400,3),np.uint8))
        browser=p.chromium.launch();page=browser.new_page(accept_downloads=True)
        page.goto('http://localhost:8511')
        expect(page.get_by_role('heading',name='HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE — ALPR')).to_be_visible(timeout=90000)
        def upload(paths):
            page.locator('input[type=file]'+('[multiple]' if isinstance(paths,list) else ':not([multiple])')).set_input_files(paths)
            page.get_by_role('button',name='Chạy nhận diện').click()
        upload(str(real));expect(page.get_by_role('heading',name='Chuỗi dự đoán')).to_be_visible(timeout=120000)
        page.get_by_text('So sánh thuật toán',exact=True).click()
        expect(page.get_by_role('heading',name='Chuỗi dự đoán')).to_be_hidden(timeout=30000)
        upload(str(real));expect(page.get_by_text('Đồng thuận là số phiếu',exact=False)).to_be_visible(timeout=90000)
        for k in ('svm','rf','cnn','ensemble'):expect(page.get_by_text(f'"{k}"',exact=True).first).to_be_visible()
        page.get_by_text('Hàng loạt',exact=True).click();upload([str(real),str(bad),str(blank)])
        expect(page.get_by_text('broken.png:',exact=False)).to_be_visible(timeout=90000)
        with page.expect_download() as event:page.get_by_role('button',name='Tải CSV').click(timeout=90000)
        csv=event.value.path().read_text(encoding='utf-8-sig')
        assert all(k in csv for k in ('14.jpg','broken.png','blank.png','Không thấy biển')),csv
        assert not page.get_by_test_id('stException').count()
        page.screenshot(path=ROOT/'reports/alpr_demo_browser.png');browser.close()
    (ROOT/'reports/alpr_demo_browser.json').write_text(json.dumps({'single_real':'PASS','compare_four':'PASS','batch_csv':'PASS','corrupt_file':'PASS','blank_large':'PASS'},indent=2))


if __name__=='__main__':main()