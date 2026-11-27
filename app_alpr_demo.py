"""Isolated ALPR demo. Keeps production and label stores untouched."""
import os
from pathlib import Path
import subprocess

import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from demo_alpr_core import History,csv_bytes,decode
from alpr import demo_service
from alpr.demo_ui import draw_ordered_boxes, show
from demo_fusion import FusionHeads, agreement
from dual_yolo_client import ROOT
from dual_yolo_pipeline import MixClassifiers
from final_ocr_dataset import read

TITLE='HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE — ALPR'
METHODS={'YOLO Character':'character_yolo','OpenCV baseline':'baseline','border_components':'border_components'}
BACKENDS={'CNN Mix':'cnn','SVM Mix':'svm','Random Forest Mix':'rf','Majority Ensemble':'ensemble',
          'SVM Fusion — thử nghiệm':'svm_fusion','Random Forest Fusion — thử nghiệm':'rf_fusion'}
ERRORS=(OSError,ValueError,RuntimeError,subprocess.SubprocessError,Image.DecompressionBombError,Image.DecompressionBombWarning,cv2.error)


@st.cache_resource
def models() -> MixClassifiers:
    return MixClassifiers()


@st.cache_resource
def fusion() -> FusionHeads:
    return FusionHeads(models().cnn)


def infer(image, method, backend):
    """Compatibility entrypoint used by Deep Analysis and regression scripts."""
    return demo_service.infer(image, method, backend, models(), fusion)


record = demo_service.result_row


def download(rows,name):st.download_button('Tải CSV',csv_bytes(rows),file_name=name,mime='application/octet-stream')


def dashboard():
    validation=read(ROOT/'experiments/yolo_character_round1/comparison/results.json')['summary']['yolo']
    st.subheader('Validation — 12 biển development, crop YOLO Plate')
    st.dataframe(pd.DataFrame([{'Tách ký tự':m,'Backend':b,'Exact':v['exact'],'CER':v['cer']}
                              for m,metrics in validation.items() for b,v in metrics['ocr'].items()]),hide_index=True)
    st.bar_chart(pd.DataFrame({m:{b:v['exact']/12 for b,v in metrics['ocr'].items()} for m,metrics in validation.items()}))
    st.bar_chart(pd.DataFrame({m:{b:v['cer'] for b,v in metrics['ocr'].items()} for m,metrics in validation.items()}))
    st.caption('Biểu đồ lần lượt: tỷ lệ exact, CER. Không phải kết quả test độc lập.')
    profile=read(ROOT/'reports/dual_yolo_profile.json');st.json({k:profile[k] for k in ('legacy_wall','cold','warm_cnn','warm_all')},expanded=False)
    path=ROOT/'reports/alpr_demo_pilot.json'
    st.subheader('Pilot — đã từng xuất hiện trong diagnostics')
    if path.is_file() and path.stat().st_size:
        try:
            pilot=read(path)
            st.dataframe(pd.DataFrame(pilot['summary']),hide_index=True)
        except (OSError,ValueError,KeyError):st.info('Artifact pilot chưa hoàn chỉnh hoặc không đọc được.')
    else:st.info('Chưa có kết quả pilot hoàn chỉnh.')
    st.warning('New10: chưa có benchmark đủ điều kiện. Không hiển thị chỉ số chưa đo.')


def main():
    st.set_page_config(page_title=TITLE,layout='wide')
    st.title(TITLE)
    st.caption('Demo thử nghiệm • RTX CUDA cho detector, OCR trên CPU • Chỉ xử lý biển có score cao nhất trong mỗi ảnh.')
    mode=st.sidebar.radio('Chức năng',['Ảnh đơn','So sánh thuật toán','Hàng loạt','Lịch sử','Đánh giá'])
    history=History(os.environ.get('ALPR_DEMO_HISTORY',str(ROOT/'data/alpr_demo_history.sqlite3')) )
    if mode=='Đánh giá':dashboard();return
    if mode=='Lịch sử':
        rows=history.search(st.text_input('Tìm tên ảnh hoặc biển số'));st.dataframe(rows)
        if rows:download(rows,'alpr-history.csv')
        confirm=st.checkbox('Xác nhận xóa toàn bộ lịch sử demo')
        if st.button('Xóa lịch sử',disabled=not confirm):history.clear();st.rerun()
        return
    method=METHODS[st.sidebar.selectbox('Tách ký tự',list(METHODS))]
    backend=BACKENDS[st.sidebar.selectbox('Nhận dạng',list(BACKENDS))]
    save=st.sidebar.checkbox('Lưu lịch sử (không lưu ảnh)')
    batch=mode=='Hàng loạt'; compare=mode=='So sánh thuật toán'
    collect=not batch and st.checkbox('Lưu ảnh lỗi để nghiên cứu (lưu pixel, không tự đưa vào train)')
    expected=st.text_input('Text đúng nếu biết (tùy chọn)') if collect else ''
    uploads=st.file_uploader('Chọn ảnh JPG/PNG',type=['jpg','jpeg','png'],accept_multiple_files=batch)
    files=uploads if batch else [uploads] if uploads else []
    st.caption('Tối đa 20 MiB / 24 MP mỗi ảnh; tối đa 20 ảnh mỗi lượt.')
    if not st.button('Chạy nhận diện',disabled=not files):return
    if len(files)>20:st.error('Chọn không quá 20 ảnh mỗi lượt.');return
    rows=[];progress=st.progress(0) if batch else None
    for i,file in enumerate(files):
        try:
            payload = file.getvalue()
            image=decode(payload)
            if not batch:show(image,'Ảnh gốc')
            requested = demo_service.request_backend(backend, compare=compare, collect=collect)
            with st.spinner('Đang xử lý…'):result=infer(image,method,requested)
            rows.extend(record(file.name,method,k,result) for k in (result['text'] if compare else [backend]))
            if collect:
                folder = demo_service.save_research_case(
                    Path(os.environ.get('ALPR_FAILURE_ROOT', ROOT/'data/alpr_failure_cases')),
                    file.name, payload, image, result, method, expected,
                )
                st.success(f'Đã lưu case nghiên cứu: {folder.name}')
            if not batch:render(image,result)
        except ERRORS as exc:
            rows.append(demo_service.error_row(file.name, method, backend, exc))
            st.error(f'{Path(file.name).name}: {exc}. Nếu là lỗi GPU/checkpoint, kiểm tra log worker rồi thử lại; không chuyển CPU.')
        if progress:progress.progress((i+1)/len(files))
    if save:history.save(rows)
    st.dataframe(rows,hide_index=True);download(rows,'alpr-results.csv')


def render(image,result):
    plate=result['plate']
    if not plate:st.warning('Không phát hiện biển số ở ngưỡng 0,4.');return
    overlay=image.copy();x,y,w,h=plate['bbox'];cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2)
    a,b=st.columns(2)
    with a:show(overlay,f'Box biển — score detector {plate["score"]:.3f}')
    x,y,w,h=plate['crop_box'];crop=image[y:y+h,x:x+w].copy(); glyphs=[]
    votes,trio=agreement(result['text'])
    draw_ordered_boxes(crop, result['glyph_boxes'])
    for i,(gx,gy,gw,gh) in enumerate(result['glyph_boxes']):
        symbols={k:p[i] for k,p in result['text'].items() if i<len(p)}
        glyphs.append({'Thứ tự':i+1,'Box':str([gx,gy,gw,gh]),**symbols,
                       'Đồng thuận':f'{votes[i]}/3' if i<len(votes) else None,
                       'Score box':result['glyph_scores'][i] if result['glyph_scores'] else None})
    with b:show(crop,'Crop và thứ tự ký tự')
    st.subheader('Chuỗi dự đoán');st.write(result['text']);st.dataframe(glyphs)
    if trio:
        st.caption(f'Đồng thuận là số phiếu, không phải xác suất đúng. So sánh: {" / ".join(trio)}.')
    disagree=[i for i,v in enumerate(votes,1) if v<3]
    if disagree:
        st.warning(f'Không chắc chắn: bất đồng {len(disagree)}/{len(votes)} vị trí ({", ".join(map(str,disagree))}). Kiểm tra ảnh gốc; không tự sửa chuỗi.')
    if not glyphs:st.warning('Có vùng biển nhưng không tách được ký tự.')
    st.json(result['timings'],expanded=False)
    st.caption(f"Worker PID {result['worker']['pid']} · {result['worker']['gpu']} · {'cold' if result['cold'] else 'warm'} · {result['timings']['total']:.3f} s")


if __name__=='__main__':main()