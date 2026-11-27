"""Opt-in deep diagnostics, leaving the Fast V1 application untouched."""
import streamlit as st

from alpr.demo_ui import draw_ordered_boxes, show
from alpr_recovery import analyze, pixels, worker
from app_alpr_demo import ERRORS, infer, models, render
from demo_alpr_core import decode


@st.cache_resource
def client():
    import atexit
    value=worker();atexit.register(value.close)
    return value


def main():
    st.title('ALPR — phân tích độ chính xác')
    mode=st.radio('Chế độ',['Fast V1','Deep Analysis Experimental'])
    st.caption('Một biển mỗi ảnh. Deep giữ box 640/640 và ngưỡng V1; tối đa 5 biến thể glyph, ngân sách mềm 12 giây, timeout IPC riêng. Không tự lưu ảnh.')
    f=st.file_uploader('Ảnh JPG/JPEG/PNG',type=['jpg','jpeg','png'])
    if not st.button('Phân tích',disabled=f is None):return
    try:
        im=decode(f.getvalue());show(im,'Ảnh gốc')
        if mode=='Fast V1':render(im,infer(im,'character_yolo','cnn'));return
        r,rois=analyze(im,client(),models())
        if not r['plate']:st.warning('Không phát hiện biển tại ngưỡng 0,4.');return
        x,y,w,h=r['plate']['crop_box']; crop=im[y:y+h,x:x+w].copy()
        draw_ordered_boxes(crop, [g['bbox'] for g in r['glyphs']], font_scale=.35, min_text_y=8)
        show(crop,'Crop và thứ tự box glyph')
        st.write('Raw CNN:',r['raw'],'Ứng viên:',r['candidate'])
        if r['uncertain_positions']:st.warning(f"Vị trí bất đồng hoặc không ổn định: {r['uncertain_positions']}. Đây không phải nhãn đúng đã xác minh.")
        if len(r['plate_rows'])>1:st.warning('Có nhiều ứng viên biển; chỉ xét box có score cao nhất.')
        if any(min(g['bbox'][0],g['bbox'][1],w-g['bbox'][0]-g['bbox'][2],h-g['bbox'][1]-g['bbox'][3])<=1 for g in r['glyphs']):st.warning('Box glyph sát mép crop; kiểm tra thiếu nét.')
        if not rois:st.warning('Không tìm thấy box ký tự.')
        if r['budget_exhausted']:st.info('Đã dừng biến thể tại ngân sách thời gian mềm.')
        st.json(r['runs']);st.json(r['top_k'],expanded=False)
        st.caption('Top-k là điểm nội bộ từng model, không phải xác suất đã hiệu chỉnh.')
        if rois:
            cols=st.columns(min(6,len(rois)))
            for i,roi in enumerate(rois):cols[i%len(cols)].image(pixels(roi,'raw'),caption=f'Glyph {i+1}',width=64)
        st.caption(f"Tổng {r['seconds']:.3f} s · YOLO {r['gpu_seconds']:.3f} s · CUDA peak reserved {r['peak_reserved']/2**20:.0f} MiB")
    except ERRORS as exc:st.error(str(exc))


if __name__=='__main__':main()