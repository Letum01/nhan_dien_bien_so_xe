"""Round2 browser navigation smoke; temporary DB, never reviewed real labels."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'reports/round2_plan/browser';output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        env={**os.environ,'ALPR_ANNOTATION_DB':str(Path(temp)/'test.db'),
             'ALPR_ANNOTATION_ROUND':'2','ALPR_ROUND_ONLY':'0'}
        with (output/'server.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','streamlit','run',str(ROOT/'label_detection.py'),
                                   '--server.port=8894','--server.headless=true','--browser.gatherUsageStats=false'],
                                   env=env,cwd=ROOT,stdout=log,stderr=log)
            try:
                for _ in range(60):
                    try:urllib.request.urlopen('http://127.0.0.1:8894/_stcore/health',timeout=1);break
                    except OSError:time.sleep(.5)
                with sync_playwright() as p:
                    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1450,'height':1100})
                    page.goto('http://127.0.0.1:8894')
                    page.get_by_text('Tổng ứng viên',exact=True).wait_for(timeout=60000)
                    assert '120' in page.locator('[data-testid="stMetricValue"]').first.inner_text()
                    page.get_by_role('button',name='Ảnh tiếp theo →',exact=True).click();page.wait_for_timeout(1600)
                    page.get_by_role('button',name='← Ảnh trước',exact=True).click();page.wait_for_timeout(1600)
                    assert not page.locator('[data-testid="stException"]').count()
                    page.screenshot(path=str(output/'round2.png'),full_page=True)
                    (output/'results.json').write_text(json.dumps({'round2_render':'PASS','candidate_count':120,
                        'previous_next':'PASS','exceptions':0,'database':'temporary','real_annotations_modified':False,
                        'approve_auto_next':'NOT TESTED in browser; no real review fabricated'},indent=2),encoding='utf-8')
                    browser.close();print('Round2 Chromium render/navigation PASS')
            finally:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                proc.wait(timeout=10)


if __name__=='__main__':main()