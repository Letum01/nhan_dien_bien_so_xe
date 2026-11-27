"""Local, explicitly consented collection on a separate Streamlit port."""
import atexit
import json
import os

import cv2
import numpy as np
import streamlit as st

from learning_center import Feedback, SPLITS, dashboard, priority
from final_ocr_dataset import ROOT


@st.cache_resource
def teacher():
    from learning_center_teacher import Teacher
    service=Teacher(); atexit.register(service.worker.close); return service


@st.cache_data(ttl=60)
def references():
    from learning_center_teacher import references as load
    return load()


def main():
    st.title('ALPR Learning Center')
    st.caption('Nghiên cứu local. Chỉ lưu khi đồng ý. Không nhập danh tính, địa chỉ, GPS hoặc thông tin chủ xe.')
    st.warning('TRAINING BLOCKED — chưa có benchmark và kiểm định alignment độc lập. Không huấn luyện hoặc đổi model.')
    db=Feedback(os.environ.get('ALPR_LEARNING_DB',str(ROOT/'data/alpr_learning_center.sqlite3')))
    with st.form('intake',clear_on_submit=True):
        f=st.file_uploader('Ảnh nghiên cứu JPG/PNG',type=['jpg','jpeg','png'])
        source=st.text_input('Nguồn ảnh (không nhập thông tin cá nhân)'); group=st.text_input('Source group đã biết')
        day=st.text_input('Ngày chụp/thu thập nếu biết (YYYY-MM-DD)'); rights=st.selectbox('Quyền sử dụng',['unknown','research_only','research_and_training'])
        split=st.selectbox('Split dự kiến',SPLITS)
        consent=st.checkbox('Tôi đồng ý lưu ảnh này để nghiên cứu local')
        submitted=st.form_submit_button('Lưu ảnh nghiên cứu')
    if submitted:
        try:
            if not f or not consent:raise ValueError('Chọn ảnh và đồng ý lưu trước khi tiếp tục.')
            db.refs=tuple(references())
            key=db.add(f.getvalue(),metadata=dict(source=source,group=group,capture_date=day,rights=rights,split=split),consent=consent)
            st.session_state['learning_selected']=key
            if split!='test':
                try: db.set_prediction(key,teacher().run(cv2.imdecode(np.frombuffer(db.pixels(key),np.uint8),cv2.IMREAD_COLOR)))
                except Exception as exc:st.warning(f'Đã lưu ảnh, chưa lưu dự đoán: {exc}')
            st.success('Đã lưu ảnh nghiên cứu.')
        except Exception as exc:
            msg=str(exc)
            if msg.startswith(('Ảnh trùng SHA-256','Ảnh gần trùng pHash')):
                stats=st.session_state.setdefault('learning_duplicates',{'exact':0,'near':0})
                stats['exact' if msg.startswith('Ảnh trùng') else 'near']+=1
            st.error(msg)
    rows=db.samples(); st.subheader('Dashboard')
    st.json({**dashboard(rows),'duplicate_rejections_this_session':st.session_state.get('learning_duplicates',{'exact':0,'near':0}),
             'verified_independent_groups':'NOT_VERIFIED'},expanded=False)
    if not rows:st.info('Chưa có mẫu nghiên cứu.');return
    # One row per group in each round: priority within each group, then interleave.
    buckets={}
    for r in sorted(rows,key=lambda r:(r['text'] is not None,r['status']=='AMBIGUOUS',-priority(r['raw']),r['id'])):
        buckets.setdefault(r['metadata']['group'] or 'UNKNOWN',[]).append(r)
    queue=[b[i] for i in range(max(map(len,buckets.values()))) for b in buckets.values() if i<len(b)]
    by={r['id']:r for r in queue}; ids=list(by); selected=st.session_state.get('learning_selected'); index=ids.index(selected) if selected in by else 0
    key=st.selectbox('Hàng đợi ưu tiên — chọn ảnh bằng bàn phím',ids,index=index,format_func=lambda k:f"{k[:10]} · {by[k]['status']} · {by[k]['metadata']['group'] or 'UNKNOWN'}")
    r=by[key]; raw=r['raw']; md=r['metadata']; data=db.pixels(key)
    st.image(data,caption='Ảnh đã bỏ metadata nhúng'); st.caption(f"Revision {r['revision']} · {r['status']} · {md['provenance']} · split {md['split']}")
    st.json(md,expanded=False)
    if md['split']!='test':
        if not raw and st.button('Chạy phân tích',key=f'infer_{key}'):
            try: db.set_prediction(key,teacher().run(cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)));st.rerun()
            except Exception as exc:st.error(str(exc))
        if raw.get('crop_box'):
            x,y,w,h=raw['crop_box']; im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)[y:y+h,x:x+w]; st.image(im,channels='BGR',caption='Crop biển đề xuất')
            st.write(raw['text']); st.caption('Top-k là thứ hạng, không phải xác suất đã hiệu chuẩn.')
            for i,g in enumerate(raw['glyphs']):
                a,b,c,d=g['bbox']; left,right=st.columns([1,4]); left.image(im[b:b+d,a:a+c],channels='BGR',caption=f'Glyph {i+1}',width=64);right.write(g['top_k'])
        elif raw:st.info('Không phát hiện biển.')
    else:st.info('Test dự kiến: không chạy hoặc hiển thị dự đoán. Chưa phải benchmark đóng băng.')
    with st.form(f'confirm_{key}_{r["revision"]}'):
        text=st.text_input('Text thật (không tự điền dự đoán)',value=r['text'] or '')
        checked=st.checkbox('Tôi xác nhận đã đọc text từ ảnh'); save=st.form_submit_button('Lưu xác nhận',type='primary'); skip=st.form_submit_button('Bỏ qua ảnh không rõ')
    if save or skip:
        try:db.confirm(key,text,confirmed=checked,skip=skip);st.rerun()
        except ValueError as exc:st.error(str(exc))
    if r['reason']:st.caption(r['reason'])
    if st.checkbox('Cho phép xuất metadata nghiên cứu',key=f'export_{key}'):
        st.download_button('Tải JSON (không gồm pixel)',json.dumps(db.export(key,allowed=True),ensure_ascii=False,indent=2),file_name=f'feedback-{key[:10]}.json',mime='application/json')
    if st.checkbox('Tôi muốn xóa ảnh và toàn bộ lịch sử mẫu này',key=f'del_{key}') and st.button('Xóa mẫu',key=f'go_{key}'):
        db.delete(key,confirmed=True);st.rerun()


if __name__=='__main__':main()