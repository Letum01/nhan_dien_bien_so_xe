"""Focused OCR/Glyph UI reusing canvas editing; never saves detection revisions."""
import copy
import hashlib
import json
import os
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from annotation_store import AnnotationStore
from annotation_ui import glyph_editor
from glyph_review import ROOT,plates_from_detection,validate_glyph_plate,normalized_text,export_glyph_dataset


VALIDATION_QUEUE = 'OCR Validation — hoàn thiện'
TEST_QUEUE = 'OCR Test — đánh giá độc lập'


def validation_queue(entries, reviewed, split='val'):
    """Pending reviewed-parent validation plates only, with confirmed text first."""
    done={r['image_id']:r for r in reviewed if r['split']==split}
    candidates=[r for r in entries if r['split']==split]
    def current(parent):
        row=done.get(parent['image_id'],{})
        return all(row.get(k)==parent[k] for k in ('detection_revision','plate_bbox','source_group','split'))
    completed=[done[r['image_id']] for r in candidates if current(r)]
    pending=sorted((r for r in candidates if not current(r)),key=lambda r:(not r['text_confirmed'],r['image_path'],r['plate_index']))
    return pending,completed


def run():
    st.title('Gắn nhãn OCR / Glyph')
    detections=AnnotationStore(Path(os.environ.get('ALPR_ANNOTATION_DB',str(ROOT/'data/annotations.sqlite3'))))
    glyphs=AnnotationStore(Path(os.environ.get('ALPR_GLYPH_DB',str(ROOT/'data/ocr_glyphs.sqlite3'))))
    entries=plates_from_detection(detections)
    if not entries:st.info('Chưa có Detection reviewed để duyệt OCR');return
    latest=glyphs.all_latest()
    reviewed=[r for r in latest.values() if r.get('review_status')=='reviewed']
    counts=[len(entries),sum(r['text_confirmed'] for r in entries),len(reviewed),sum(len(r['characters']) for r in reviewed)]
    for col,label,value in zip(st.columns(4),['Biển Detection reviewed','Text confirmed (detection metadata)','Biển Glyph reviewed','Glyph reviewed'],counts):col.metric(label,value)
    st.caption('Kho OCR riêng có revision/draft. Group/split kế thừa nguồn, không đổi Detection reviewed. OpenCV không tạo nhãn class.')
    pilot=ROOT/'reports/glyph_pilot.json'
    pilot_ids=set(json.loads(pilot.read_text(encoding='utf-8'))['plate_ids']) if pilot.exists() else set()
    from glyph_progress import summarize
    progress=summarize(detections,glyphs,pilot_ids,reviewed_snapshot=reviewed)
    import pandas as pd
    default_queue={'train':1,'hard':2,'validation':3,'test':4}.get(os.environ.get('ALPR_OCR_QUEUE'),0)
    queue_mode=st.selectbox('Hàng đợi OCR',['Tất cả / Ưu tiên validation','OCR Train — ưu tiên','OCR Hard Cases — ưu tiên',VALIDATION_QUEUE,TEST_QUEUE],index=default_queue)
    train_priority=queue_mode=='OCR Train — ưu tiên'
    hard_priority=queue_mode=='OCR Hard Cases — ưu tiên'
    validation_priority=queue_mode==VALIDATION_QUEUE
    test_priority=queue_mode==TEST_QUEUE
    queue_locked=train_priority or hard_priority or validation_priority or test_priority
    priority_entries=None
    if test_priority:
        priority_entries,completed=validation_queue(entries,reviewed,'test')
        st.subheader(TEST_QUEUE)
        st.warning('Ba biển pilot đã từng được đánh giá detector, không phải final test độc lập. Gắn nhãn thủ công, không hiển thị dự đoán hoặc proposal.')
        st.write(f'Còn {len(priority_entries)} / đã duyệt {len(completed)} biển test.')
    if validation_priority:
        priority_entries,completed=validation_queue(entries,reviewed)
        st.subheader(VALIDATION_QUEUE)
        for col,label,value in zip(st.columns(3),['Validation còn lại','Validation đã duyệt','Glyph validation đã duyệt'],
                                  [len(priority_entries),len(completed),sum(len(r['characters']) for r in completed)]):
            col.metric(label,value)
        with st.expander('Danh sách biển validation cần duyệt',expanded=True):
            st.dataframe(pd.DataFrame([{'Ảnh':Path(r['image_path']).name,'Source group':r['source_group'],'Split':r['split']} for r in priority_entries]),hide_index=True)
    if hard_priority:
        from ocr_hard_review import load_hard_queue
        hard=load_hard_queue(detections,glyphs,reviewed_snapshot=reviewed)
        st.subheader('OCR Hard Cases — ưu tiên')
        train_rows=[r for r in reviewed if r['split']=='train']
        for col,label,value in zip(st.columns(3),['Hard cases còn lại','Hard cases hoàn thành','Tổng glyph train'],
                                   [len(hard['pending']),len(hard['completed']),sum(len(r['characters']) for r in train_rows)]):col.metric(label,value)
        st.caption('Ưu tiên nhỏ/mờ/nghiêng/sát viền là proxy chưa xác minh, không phải nhãn thật. Không dùng model để gán class.')
        counts_train={c:sum(ch.get('confirmed') and ch.get('label')==c for r in train_rows for ch in r['characters']) for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
        st.caption('Class thiếu train: '+', '.join(c for c,n in counts_train.items() if not n))
        st.caption('Class train chỉ 1–2 mẫu: '+', '.join(c for c,n in counts_train.items() if 1<=n<=2))
        if hard['errors']:
            st.error('Queue có parent lỗi; mục lỗi không được hiển thị để sửa nhãn âm thầm.')
            st.json(hard['errors'])
        priority_entries=hard['pending']
        for milestone in (10,15):
            if len(hard['completed'])>=milestone and not glyphs.preference(f'hard{milestone}_audit'):
                from audit_glyph_reviews import main as audit
                folder,audit_report=audit(glyphs.path,detections.path)
                from summarize_glyph_milestone import summarize_folder
                if audit_report['export_validator'].get('status')=='PASS':summarize_folder(folder)
                glyphs.set_preference(f'hard{milestone}_audit',{'folder':str(folder),'validator':audit_report['export_validator']})
            if glyphs.preference(f'hard{milestone}_audit'):st.info(f'Mốc hard {milestone}: '+str(glyphs.preference(f'hard{milestone}_audit')))
    if train_priority:
        from ocr_train_queue import load_queue
        try:document,priority_entries=load_queue(detections)
        except (ValueError,FileNotFoundError) as exc:st.error(str(exc));return
        queue_ids={r['image_id'] for r in priority_entries}
        completed=[r for r in reviewed if r['image_id'] in queue_ids]
        train_rows=[r for r in reviewed if r['split']=='train']
        class_counts={c:sum(ch.get('confirmed') and ch.get('label')==c for r in train_rows for ch in r['characters']) for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
        st.subheader('OCR Train — ưu tiên')
        st.write(f"Hàng đợi: {len(completed)}/15 biển mới hoàn thành; train toàn corpus: {len(train_rows)} biển / {sum(len(r['characters']) for r in train_rows)} glyph")
        st.progress(len(completed)/15)
        st.caption('Thiếu train: '+', '.join(c for c,n in class_counts.items() if n==0))
        st.caption('Train ít mẫu (1–2): '+', '.join(c for c,n in class_counts.items() if 1<=n<=2))
        st.dataframe(pd.DataFrame([{'class':c,'train':n,'validation':sum(ch.get('confirmed') and ch.get('label')==c for r in reviewed if r['split']=='val' for ch in r['characters'])} for c,n in class_counts.items()]),hide_index=True)
        if len(completed)==15:
            signature=hashlib.sha256(json.dumps([(r['image_id'],r['annotation_version']) for r in sorted(reviewed,key=lambda r:r['image_id'])]).encode()).hexdigest()
            if glyphs.preference('train15_signature')!=signature:
                from audit_glyph_reviews import main as audit
                folder,audit_report=audit(glyphs.path,detections.path)
                from summarize_glyph_milestone import summarize_folder
                if audit_report['export_validator'].get('status')=='PASS':summarize_folder(folder)
                glyphs.set_preference('train15_signature',signature)
                glyphs.set_preference('train15_audit',{'folder':str(folder),'validator':audit_report['export_validator']})
            st.info('15 biển mới đã hoàn tất: '+str(glyphs.preference('train15_audit')))
    st.subheader('Tiến độ glyph theo split nguồn')
    st.dataframe(pd.DataFrame(progress['split_progress']),hide_index=True)
    st.dataframe(pd.DataFrame([{'class':c,'samples':n} for c,n in progress['class_counts'].items()]),hide_index=True)
    st.caption('Class chưa có: '+', '.join(progress['missing_classes']))
    st.caption('Class ít mẫu (<5, chỉ ưu tiên gắn nhãn): '+', '.join(progress['low_sample_classes']))
    if progress['edge_warnings']:
        st.warning(f"{len(progress['edge_warnings'])} cảnh báo nét/mép heuristic: không đồng nghĩa glyph sai; crop sát nét thường chạm biên. Xem contact sheet trước khi sửa.")
        with st.expander('Chi tiết glyph cần xem mép/nét'):st.dataframe(pd.DataFrame(progress['edge_warnings']),hide_index=True)
    st.caption(f"Còn {progress['remaining_all']} biển toàn nguồn, {progress['remaining_pilot']} trong pilot cũ. Không tự đổi split để tạo validation.")
    if len(reviewed)>=10:
        signature=hashlib.sha256(json.dumps([(r['image_id'],r['annotation_version']) for r in sorted(reviewed,key=lambda r:r['image_id'])]).encode()).hexdigest()
        if glyphs.preference('milestone10_signature')!=signature:
            from audit_glyph_reviews import main as audit
            folder,audit_report=audit(glyphs.path,detections.path)
            glyphs.set_preference('milestone10_signature',signature)
            glyphs.set_preference('milestone10_audit',{'folder':str(folder),'validator':audit_report['export_validator']})
        st.info('Mốc >=10 biển: audit/validator '+str(glyphs.preference('milestone10_audit')))
    validation_first=st.checkbox('Ưu tiên validation độc lập chưa reviewed',value=not queue_locked,disabled=queue_locked)
    pilot_only=st.checkbox('Chỉ pilot OCR 20 biển',value=False,disabled=queue_locked)
    if queue_locked:
        entries=priority_entries
        validation_first=False;pilot_only=False
    if pilot_only and pilot.exists():
        ids=set(json.loads(pilot.read_text(encoding='utf-8'))['plate_ids'])
        entries=[r for r in entries if r['image_id'] in ids]
    if validation_first:
        priorities={r['image_id']:i for i,r in enumerate(progress['queue'])}
        entries.sort(key=lambda r:priorities.get(r['image_id'],len(priorities)+1))
    if not entries:st.info('Không có biển trong bộ lọc');return
    state=st.session_state
    def persist():
        if state.get('ocr_work'):
            glyphs.save_draft(state['ocr_work'],state['ocr_base'])
    def switch():
        try:persist()
        except ValueError as exc:
            state['ocr_error']=str(exc);state['ocr_selected']=state['ocr_active']
    ids=[r['image_id'] for r in entries];parents={r['image_id']:r for r in entries}
    if state.get('ocr_queue_mode')!=queue_mode:
        if state.get('ocr_work'):
            try:persist()
            except ValueError as exc:st.error(str(exc));return
        state['ocr_selected']=next((identity for identity in ids if identity not in {r['image_id'] for r in reviewed}),ids[0])
        state['ocr_queue_mode']=queue_mode
    if 'ocr_selected' not in state or state['ocr_selected'] not in ids:
        remembered=glyphs.preference('last_plate',ids[0]);state['ocr_selected']=ids[0] if validation_first else remembered if remembered in ids else ids[0]
    selected=st.selectbox('Biển cần gắn glyph',ids,format_func=lambda identity:f"{Path(parents[identity]['image_path']).name} / plate {parents[identity]['plate_index']} / {parents[identity]['split']} / text {'confirmed' if parents[identity]['text_confirmed'] else 'chưa duyệt'}",key='ocr_selected',on_change=switch)
    if state.get('ocr_error'):st.error(state.pop('ocr_error'));return
    parent=parents[selected]
    if hard_priority:st.json({'Lý do ưu tiên (heuristic)':parent['hard_priority']})
    if state.get('ocr_active')!=selected:
        old=glyphs.latest(selected) or {};draft=glyphs.draft(selected)
        base=old.get('annotation_version',0)
        if draft and draft[1]!=base:st.error('Draft version conflict; không ghi đè');return
        working=copy.deepcopy(draft[0] if draft else old or parent)
        for field in ('source_image_id','plate_bbox','source_group','split','detection_revision'):
            if working.get(field)!=parent.get(field):st.error('Detection parent đã đổi; cần kiểm tra lại OCR');return
        state['ocr_work']=working;state['ocr_base']=base;state['ocr_active']=selected
        state['ocr_generation']=state.get('ocr_generation',0)+1
        glyphs.set_preference('last_plate',selected)
    row=state['ocr_work'];x,y,w,h=parent['plate_bbox']
    source=Image.open(ROOT/parent['image_path']).convert('RGB');crop=source.crop((x,y,x+w,y+h))
    st.image(crop,caption=f"Crop GT xywh={parent['plate_bbox']} — {parent['source_group']} / {parent['split']}",width=440)
    row['plate_text']=st.text_input('Văn bản đúng bằng người (giữ nguyên)',value=row.get('plate_text',''),key='ocr_text_'+selected)
    row['unreadable']=st.checkbox('Không đọc được / thiếu nét',value=row.get('unreadable',False),key='ocr_unreadable_'+selected)
    row['layout']=st.selectbox('Bố cục dòng',['UNKNOWN','one_line','two_lines'],index=['UNKNOWN','one_line','two_lines'].index(row.get('layout','UNKNOWN')),key='ocr_layout_'+selected)
    text_key=hashlib.sha256((row['plate_text']+str(row['unreadable'])).encode()).hexdigest()[:12]
    row['text_confirmed']=st.checkbox('Tôi xác nhận text khớp crop',value=bool(parent['text_confirmed'] and row['plate_text']==parent['plate_text']),key='ocr_text_confirm_'+selected+text_key)
    st.caption('Xóa contour viền/ốc vít bằng index; vẽ glyph thiếu bằng rect; transform để resize/move; split/merge làm mất nhãn xác nhận. Dòng 0 trên, 1 dưới.')
    try:
        glyph_editor(row,crop,f'ocr_{selected}_{state["ocr_generation"]}',persist,allow_proposals=parent['split']!='test')
        text=normalized_text(row['plate_text'])
        observed=''.join(c.get('label','') for c in row['characters'])
        edge_indices=[i for i,c in enumerate(row['characters'])
                      if min(c['bbox'][0],c['bbox'][1],w-c['bbox'][0]-c['bbox'][2],h-c['bbox'][1]-c['bbox'][3])<=2]
        if edge_indices:st.warning(f'Glyph sát mép crop {edge_indices}: kiểm tra nét; cảnh báo không tự xóa/đổi nhãn.')
        st.write('Thứ tự glyph hiện tại:',observed)
        st.caption(f"Text {len(text)} ký tự / proposal {len(row['characters'])}; không tự điền nhãn theo text.")
        errors=[]
        try:validate_glyph_plate(row,w,h)
        except ValueError as exc:errors.append(str(exc));st.warning(str(exc))
        content=hashlib.sha256(json.dumps(row['characters'],sort_keys=True).encode()+row['plate_text'].encode()).hexdigest()[:12]
        confirm=st.checkbox('Tôi đã kiểm tra từng glyph và thứ tự; xác nhận Glyph reviewed',key='ocr_final_'+selected+content)
        if st.button('Lưu Glyph reviewed',disabled=bool(errors) or not confirm):
            saved=glyphs.save({**row,'review_status':'reviewed','confirmed_by_user':True},state['ocr_base'])
            row.update(saved);state['ocr_base']=saved['annotation_version'];persist()
            st.success('Đã lưu OCR revision; Detection/split không đổi')
            st.rerun()
        persist()
    except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
    with st.expander('Export glyph reviewed — thư mục mới'):
        destination=st.text_input('Output glyph dataset',value='datasets/glyph_round01')
        if st.button('Export dữ liệu glyph đã duyệt'):
            try:
                count=export_glyph_dataset(glyphs,detections,ROOT/destination)
                st.success(f'Đã xuất {count} glyph, giữ split nguồn. Chạy validator trước training.')
            except (ValueError,FileExistsError) as exc:st.error(str(exc))


if __name__=='__main__':run()