"""Development-only inference and proposal QA; never writes human labels."""
from datetime import datetime, timezone
from pathlib import Path

import cv2
import streamlit as st

from development_autolabel import ROOT, export, pool, propose
from dual_yolo_client import shared_worker
from dual_yolo_pipeline import MixClassifiers


@st.cache_resource
def models():return MixClassifiers()


def main():
    st.title('ALPR — công cụ development')
    st.warning('Chỉ ảnh train trong danh sách đã lọc. Không dùng trang này để gắn nhãn final test.')
    mode=st.radio('Chế độ',['Nhận diện đơn','Nhận diện hàng loạt','Auto Labeling — gắn nhãn tự động','So sánh SVM/RF/CNN','Báo cáo hiệu năng'])
    if mode=='Báo cáo hiệu năng':
        from final_ocr_dataset import read
        p=ROOT/'reports/dual_yolo_profile.json'
        if p.exists():
            data=read(p);st.json({k:data[k] for k in ('legacy_wall','cold','warm_all','warm_cnn')})
        else:st.info('Chưa có báo cáo profiling.')
        return
    rows,excluded=pool(deduplicate=mode in ('Nhận diện hàng loạt','Auto Labeling — gắn nhãn tự động'))
    st.caption(f'{len(rows)} ảnh khả dụng; {len(excluded)} mục bị lọc trong chế độ này.')
    if not rows:return
    options={r['image']:r for r in rows}
    if mode=='Auto Labeling — gắn nhãn tự động' and st.button('Mở batch đề xuất đã lưu'):
        from final_ocr_dataset import read
        manifest=ROOT/'reports/automated_workflow_proposals_v1.json'
        if manifest.is_file():
            records=read(manifest)['samples']
            if all(r['image'] in options and r['sha256']==options[r['image']]['sha256'] and r['label_origin']=='AUTO_PSEUDO_LABEL' for r in records):
                st.session_state['dev_batch']=records
                st.session_state['dev_generation']=st.session_state.get('dev_generation',0)+1
            else:st.error('Manifest không còn khớp allowlist train.')
        else:st.info('Chưa có batch đề xuất.')
    if mode in ('Nhận diện đơn','So sánh SVM/RF/CNN'):
        selected=[st.selectbox('Ảnh train',list(options))]
    else:
        selected=st.multiselect('Ảnh train',list(options))
    if st.button('Chạy',disabled=not selected):
        try:
            with st.spinner('Đang chạy GPU…'):
                batch=list(propose([options[x] for x in selected],shared_worker,models()))
                st.session_state['dev_batch']=batch;st.session_state['dev_generation']=st.session_state.get('dev_generation',0)+1
        except (ValueError,OSError,RuntimeError) as exc:st.error(str(exc))
    batch=[r for r in st.session_state.get('dev_batch',[]) if r['image'] in options and options[r['image']]['sha256']==r['sha256']]
    if not batch:return
    indexes=sorted(range(len(batch)),key=lambda i:-len(batch[i]['flags']))
    index=st.selectbox('Kết quả đã xử lý',indexes,format_func=lambda i:f"{Path(batch[i]['image']).name} — {len(batch[i]['flags'])} cờ")
    row=batch[index]; result=row['result']; image=cv2.imread(str(ROOT/row['image']))
    if image is None:return
    overlay=image.copy(); plate=result['plate']; glyph_rows=[]
    if plate:
        x,y,w,h=plate['bbox'];cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2)
        px,py,_,_=plate['crop_box']
        st.caption(f"Plate detector: {plate['score']:.3f}")
        for i,((x,y,w,h),score) in enumerate(zip(result['glyph_boxes'],result['glyph_scores'])):
            cv2.rectangle(overlay,(px+x,py+y),(px+x+w,py+y+h),(0,180,255),1)
            cv2.putText(overlay,str(i+1),(px+x,py+y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,180,255),1)
            votes={k:result['text'][k][i] for k in ('svm','rf','cnn','ensemble')}
            glyph_rows.append({'glyph':i+1,'box':str([x,y,w,h]),'detector_score':score,'agreement':max(list(votes.values())[:3].count(v) for v in list(votes.values())[:3]),**votes})
    a,b=st.columns(2);a.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption='Ảnh gốc');b.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB),caption='Box đề xuất')
    st.json(result['text']);st.caption('Đồng thuận và score chỉ để ưu tiên kiểm tra, không xác nhận nhãn.')
    if glyph_rows:st.dataframe(glyph_rows)
    st.write('Cờ ưu tiên:',', '.join(row['flags']) or 'không phát hiện theo heuristic')
    if mode=='Auto Labeling — gắn nhãn tự động':
        st.info('HUMAN_REVIEWED chỉ đến từ kho nhãn người dùng. Trạng thái tại đây không thể tạo nhãn reviewed.')
        key=f"{row['sha256']}:{st.session_state['dev_generation']}"
        row['edited_text']=st.text_input('Sửa chuỗi bằng bàn phím',value=row.get('edited_text',result['text']['ensemble']),key=key+'text')
        if len(row['edited_text'])!=len(glyph_rows):st.warning('Chiều dài khác số box: không thể ánh xạ class theo vị trí.')
        states=['AUTO_PSEUDO_LABEL','NEEDS_REVIEW','REJECTED']
        row['status']=st.selectbox('Trạng thái đề xuất',states,index=states.index(row['status']),key=key+'status')
        if st.button('Xuất manifest đề xuất riêng'):
            path=ROOT/'reports'/('development_pseudo_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
            export(batch,path);st.success(str(path))


if __name__=='__main__':main()