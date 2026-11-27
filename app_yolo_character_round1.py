"""Isolated development UI; does not change the production Streamlit app."""
import subprocess
from time import perf_counter

import cv2
import numpy as np
import streamlit as st

from glyph_review import ROOT
from dual_yolo_client import shared_worker
from dual_yolo_pipeline import MixClassifiers, run

WEIGHTS=ROOT/'experiments/yolo_character_round1/weights/best.pt'
METHODS={'OpenCV baseline':'baseline','border_components':'border_components','YOLO Character':'character_yolo'}
BACKENDS={'SVM Mix':'svm','Random Forest Mix':'rf','CNN Mix':'cnn','Majority Ensemble (Mix)':'ensemble'}


@st.cache_resource
def models():
    return MixClassifiers()


def show(image, **kwargs):
    st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),**kwargs)


def main():
    st.title('YOLO Character — thử nghiệm')
    st.caption('So sánh bộ tách ký tự với cùng classifier Mix. OpenCV baseline ở đây chỉ là baseline tách ký tự; không dùng weights OCR baseline cũ. Plate YOLO 0,4; Character YOLO 0,25.')
    method=st.selectbox('Tách ký tự',list(METHODS))
    backend=st.selectbox('Nhận dạng',list(BACKENDS))
    upload=st.file_uploader('Ảnh xe (JPG/PNG)',type=['jpg','jpeg','png'])
    if not upload:
        return
    image=cv2.imdecode(np.frombuffer(upload.getvalue(),np.uint8),cv2.IMREAD_COLOR) if upload.size else None
    if image is None:
        st.error('Không đọc được ảnh.'); return
    if not st.button('Chạy thử nghiệm'):
        return
    try:
        with st.spinner('Đang phát hiện và nhận dạng…'):
            start=perf_counter(); result=run(image,shared_worker,models(),BACKENDS[backend],METHODS[method]); plate=result['plate']
            if plate is None:
                st.warning('Không phát hiện biển ở ngưỡng 0,4.'); return
            x,y,w,h=plate['crop_box']; crop=image[y:y+h,x:x+w]
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        st.error(f'Không chạy được thử nghiệm: {exc}'); return
    overlay=image.copy(); px,py,pw,ph=plate['bbox']
    cv2.rectangle(overlay,(px,py),(px+pw-1,py+ph-1),(0,255,0),2)
    show(overlay,caption=f'Plate bbox {plate["bbox"]} — score {plate["score"]:.3f}')
    vis=crop.copy(); text=result['text'][BACKENDS[backend]]; scores=result['glyph_scores']
    rows=[]
    for i,(gx,gy,gw,gh) in enumerate(result['glyph_boxes']):
        cv2.rectangle(vis,(gx,gy),(gx+gw-1,gy+gh-1),(0,255,0),1)
        cv2.putText(vis,str(i+1),(gx,max(8,gy)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
        rows.append({'Thứ tự':i+1,'BBox':str([gx,gy,gw,gh]),'Ký tự':text[i],'Score detector':round(scores[i],3) if scores else None})
    show(vis,caption='Plate crop và thứ tự ký tự')
    st.subheader('Plate text dự đoán'); st.code(text or '(trống)')
    if not text or '?' in text:
        st.warning('Kết quả không chắc chắn: không có glyph hoặc Ensemble không có đa số.')
    st.caption('Score của box không phải xác suất chuỗi biển số đúng.')
    t=result['timings']; info=result['worker']; render=perf_counter()
    st.caption(f'Worker PID {info["pid"]} · {info["gpu"]} CUDA / OCR TensorFlow CPU · '+('cold' if result['cold'] else 'warm'))
    st.write(f'Khởi tạo worker: {info["startup_seconds"]:.3f}s | Plate: {t["plate_predict"]:.3f}s | Character: {t.get("character_predict",0):.3f}s | Tổng inference: {t["total"]:.3f}s')
    st.json({k:round(v,5) for k,v in t.items()},expanded=False)
    if rows:
        st.dataframe(rows,hide_index=True)
        with st.expander('Glyph previews'):
            columns=st.columns(4)
            for i,(gx,gy,gw,gh) in enumerate(result['glyph_boxes']):
                columns[i%4].image(cv2.cvtColor(crop[gy:gy+gh,gx:gx+gw],cv2.COLOR_BGR2RGB),width=65,caption=f'{i+1}: {text[i]}')
    st.caption(f'Streamlit enqueue (sau overlay, không gồm browser paint): {perf_counter()-render:.3f}s · yêu cầu: {perf_counter()-start:.3f}s')


if __name__=='__main__':
    main()