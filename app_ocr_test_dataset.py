"""Manual test annotation only. Never imports a detector or classifier."""
import copy
import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from annotation_store import AnnotationStore, canvas_boxes
from annotation_ui import glyph_editor, rectangles
from audit_final_new10 import QUEUE
from dataset_audit import fingerprint
from dataset_validation import validate_boxes
from final_ocr_dataset import STORE, audit_candidates, pilot_rows, read, relative_image, validate_corpus, validate_manual
from final_ocr_intake import preview_bytes, upload
from glyph_review import ROOT

NEW10='Final Test — 10 ảnh mới'


def apply_plate_box(row, box, width, height):
    validate_boxes([box],width,height)
    if box!=row.get('plate_bbox'):
        row.update(plate_bbox=box,characters=[],plate_text='',text_confirmed=False,
                   confirmed_text=None,plate_box_confirmed=False)


def confirmed_source(row, known_groups):
    group=row.get('source_group','').strip().casefold()
    return bool(row.get('provenance_confirmed') and group not in {'','unknown','unverified','needs_human_confirmation',*known_groups}
                and row.get('confirmed_group')==row.get('source_group')
                and row.get('confirmed_evidence')==row.get('source_evidence')
                and len(row.get('source_evidence','').strip())>=20)


def main():
    st.title('Gắn nhãn OCR test thủ công')
    store=AnnotationStore(Path(os.environ.get('ALPR_FINAL_OCR_DB',STORE)))
    modes=['OCR Pilot Test','OCR Final Test — nguồn độc lập',NEW10]
    mode=st.radio('Hàng đợi',modes,index=2 if os.environ.get('ALPR_FINAL_OCR_QUEUE')=='new10' else 0)
    pilot=mode=='OCR Pilot Test'; cohort='pilot' if pilot else 'final'
    new10=mode==NEW10
    latest=store.all_latest()
    if pilot:
        st.warning('16.jpg, 17.jpg, 18.jpg từng xuất hiện trong diagnostics. Đây không phải final blind test. Mọi xác minh bổ sung lưu riêng.')
        entries=pilot_rows()
    else:
        st.info('UNKNOWN không phải nguồn độc lập. Hãy xác minh nhóm xe/lần chụp/video và lịch sử sử dụng trước khi xác nhận nhãn.')
        entries=[]
        audit=audit_candidates(store.path,write=False)
        pinned=set(read(QUEUE)) if new10 else set()
        for e in audit['rows']:
            if new10 and (not e.get('imported_at') or e['sha256'] not in pinned):continue
            key=hashlib.sha256(('final:'+e['image']).encode()).hexdigest()
            entries.append({'image_id':key,'image_path':e['image'],'source_sha256':e['sha256'],'source_phash':e['phash'],
                            'split':'test','source_group':e.get('metadata',{}).get('source_group') or 'UNKNOWN',
                            'source_evidence':e.get('metadata',{}).get('source_evidence',''),
                            'plate_bbox':None,'characters':[], 'audit':e})
        with st.expander('Nhập ảnh Final Test mới'):
            files=st.file_uploader('Upload JPG/PNG',type=['jpg','jpeg','png'],accept_multiple_files=True)
            previews=[]
            for f in files:
                try:
                    preview,_=preview_bytes(f.getvalue());preview.thumbnail((320,240));st.image(preview,caption=f.name)
                    previews.append(f)
                except (ValueError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:st.error(str(exc))
            options={relative_image(e['image_path']):Path(e['image_path']).name for e in entries
                     if not e['audit'].get('superseded_by') and latest.get(e['image_id'],{}).get('review_status')!='reviewed'}
            target=st.selectbox('Thay ứng viên (chỉ khi nhập một ảnh)',[None,*options],format_func=lambda k: options.get(k,'Không thay'))
            with st.form('upload_metadata'):
                metadata={k:st.text_input(label) for k,label in [('source_group','Source group nhập ảnh'),('capture','Thời điểm / nhóm lần chụp'),
                          ('vehicle','Loại phương tiện'),('conditions','Điều kiện ảnh')]}
                metadata['source_evidence']=st.text_area('Thông tin nguồn và lịch sử sử dụng')
                submitted=st.form_submit_button('Nhập ảnh vào hàng đợi')
            if submitted:
                if target and (len(files)!=1 or len(previews)!=1):st.error('Chọn đúng một ảnh hợp lệ để thay thế')
                elif not previews:st.error('Chọn ít nhất một ảnh hợp lệ')
                else:
                    messages=[]
                    for f in previews:
                        try:upload(store.path,f.name,f.getvalue(),metadata,target)
                        except (ValueError,OSError) as exc:messages.append((False,f'{f.name}: {exc}'))
                        else:messages.append((True,f'{f.name}: đã nhập, provenance UNKNOWN'))
                    st.session_state['intake_messages']=messages;st.rerun()
        for ok,message in st.session_state.pop('intake_messages',[]):
            (st.success if ok else st.error)(message)
    completed=[latest[e['image_id']] for e in entries if e.get('audit',{}).get('status')!='REPLACE' and not e.get('audit',{}).get('superseded_by') and latest.get(e['image_id'],{}).get('review_status')=='reviewed']
    existing=[e for e in entries if e.get('original_ocr_revision') and e['image_id'] not in latest] if pilot else []
    excluded=[e for e in entries if e.get('audit',{}).get('status')=='REPLACE' or e.get('audit',{}).get('superseded_by') or latest.get(e['image_id'],{}).get('review_status')=='rejected']
    labels=completed+existing
    verified=len({r['source_group'] for r in completed}) if not pilot else len(entries)
    if new10:
        verified=0
        for e in entries:
            if e in excluded:continue
            approved_row=latest.get(e['image_id'],{})
            saved=store.draft(e['image_id'])
            row_for_count=saved[0] if saved and saved[1]==approved_row.get('annotation_version',0) else approved_row
            verified+=confirmed_source(row_for_count,audit['known_groups'])
    for col,label,value in zip(st.columns(4),['Ảnh đã xác minh nguồn' if new10 else 'Nhóm đã xác minh' if not pilot else 'Biển trong pilot','Ảnh còn chờ','Biển đã reviewed','Glyph đã reviewed'],
                              [verified,
                               len(entries)-len(labels)-len(excluded),len(labels),sum(len(r['characters']) for r in labels)]):col.metric(label,value)
    if not pilot:
        active=len(entries)-len(excluded)
        st.caption(f'Tổng ứng viên: {len(entries)} · Cần thay: {sum(e["audit"]["status"]=="REPLACE" and not e["audit"].get("superseded_by") for e in entries)} · Đã loại/thay: {len(excluded)}')
        st.progress(len(completed)/active if active else 0)
    if not entries:return
    selected=st.selectbox('Ảnh cần kiểm tra', [e['image_id'] for e in entries],format_func=lambda key:next(e.get('audit',{}).get('filename') or Path(e['image_path']).name for e in entries if e['image_id']==key))
    entry=next(e for e in entries if e['image_id']==selected)
    if entry.get('audit',{}).get('superseded_by'):
        st.info('Đã thay bằng '+entry['audit']['superseded_by']+'; ảnh và lịch sử cũ được giữ.');return
    if entry.get('audit',{}).get('status')=='REPLACE':
        st.error('Cần thay thế: '+', '.join(entry['audit']['reasons']));return
    previous=latest.get(selected,{})
    if previous.get('review_status')=='rejected':
        st.info('Đã loại: '+previous.get('exclusion_reason',''));return
    base=previous.get('annotation_version',0)
    draft=store.draft(selected)
    if draft and draft[1]!=base:
        st.error('Draft cũ xung đột với revision mới. Không ghi đè.');return
    row=copy.deepcopy(draft[0] if draft else previous or entry)
    row.pop('audit',None);row.update(cohort=cohort,split='test')
    path=ROOT/entry['image_path']; image=Image.open(path).convert('RGB')
    st.image(image,caption=entry.get('audit',{}).get('filename') or Path(entry['image_path']).name,width=min(image.width,750))
    prefix=cohort+'_'+selected+'_'+str(base)
    if not pilot:
        if entry['audit'].get('metadata'):st.json(entry['audit']['metadata'],expanded=False)
        if new10:
            report=read(ROOT/'reports/final_test_new10_audit.json')
            details=next(r for r in report['rows'] if r['sha256']==entry['source_sha256'])
            st.caption(f"{details['width']} × {details['height']} px · Laplacian {details['laplacian']} · xám TB {details['mean_gray']}")
            for warning in details['quality_warnings']:st.warning(warning)
        row['source_group']=st.text_input('Source group thực tế',value=row['source_group'],key=prefix+'_group').strip()
        row['source_evidence']=st.text_area('Bằng chứng nguồn: xe, lần chụp/video và lịch sử sử dụng',value=row.get('source_evidence',''),key=prefix+'_evidence')
        evidence_key=hashlib.sha256((row['source_group']+row['source_evidence']).encode()).hexdigest()[:12]
        old_evidence=hashlib.sha256((row.get('confirmed_group','')+row.get('confirmed_evidence','')).encode()).hexdigest()[:12]
        row['provenance_confirmed']=st.checkbox('Tôi đã kiểm tra nguồn độc lập và không dùng trong phát triển',value=bool(row.get('provenance_confirmed') and evidence_key==old_evidence),key=prefix+'_provenance_'+evidence_key)
        if row['provenance_confirmed']:
            row.update(confirmed_group=row['source_group'],confirmed_evidence=row['source_evidence'])
        source_ok=confirmed_source(row,audit['known_groups'])
        st.caption('Provenance: '+('đã xác nhận bởi người duyệt' if source_ok else 'UNKNOWN — cần nhóm hợp lệ, bằng chứng và xác nhận'))
        if row['source_group'].casefold() in audit['known_groups']:st.warning('Source group trùng nhóm tham chiếu đã biết.')
        reason=st.text_input('Lý do loại khỏi corpus (không xóa ảnh)',key=prefix+'_reason')
        if st.button('Ghi nhận loại ứng viên',disabled=not reason.strip() or previous.get('review_status')=='reviewed'):
            store.save({**row,'review_status':'rejected','exclusion_reason':reason.strip()},base);store.clear_draft(selected);st.rerun()
    st.caption('Box biển: tọa độ xywh theo pixel trên ảnh gốc. Đổi crop sẽ xóa glyph trong bản làm việc, không sửa revision cũ.')
    current=row.get('plate_bbox') or [0,0,0,0]
    if st.checkbox('Mở canvas vẽ box biển',key=prefix+'_plate_canvas'):
        from streamlit_drawable_canvas import st_canvas
        scale=min(1.,750/image.width,600/image.height)
        canvas=st_canvas(background_image=image,width=round(image.width*scale),height=round(image.height*scale),
                         initial_drawing=rectangles([current] if row.get('plate_bbox') else [],scale),
                         drawing_mode=st.radio('Thao tác box biển',['rect','transform'],horizontal=True,key=prefix+'_plate_mode'),
                         stroke_color='green',stroke_width=1,key=prefix+'_plate_'+str(current))
        if st.button('Áp dụng box từ canvas',key=prefix+'_commit_canvas'):
            try:
                boxes=canvas_boxes((canvas.json_data or {}).get('objects',[]),scale,image.width,image.height)
                if len(boxes)!=1:raise ValueError('Canvas phải có đúng một box biển. Xóa box cũ trước khi vẽ lại.')
                apply_plate_box(row,boxes[0],image.width,image.height)
                store.save_draft(row,base);st.rerun()
            except ValueError as exc:st.error(str(exc))
    with st.form(prefix+'_bbox'):
        proposed=[int(col.number_input(axis,min_value=0,max_value=limit,value=v,key=prefix+axis))
                  for col,axis,limit,v in zip(st.columns(4),('x','y','w','h'),(image.width,image.height)*2,current)]
        if st.form_submit_button('Áp dụng box biển thủ công'):
            try:apply_plate_box(row,proposed,image.width,image.height)
            except ValueError as exc:st.error(str(exc))
            else:
                store.save_draft(row,base);st.rerun()
    if not row.get('plate_bbox'):
        if row.get('source_evidence') or row['source_group']!='UNKNOWN':store.save_draft(row,base)
        return
    x,y,w,h=row['plate_bbox'];crop=image.crop((x,y,x+w,y+h));st.image(crop,caption='Crop box thủ công',width=min(480,max(140,w)))
    edit_key=prefix+'_'+str(x)+'_'+str(y)+'_'+str(w)+'_'+str(h)
    row['plate_box_confirmed']=st.checkbox('Tôi xác nhận box biển',value=bool(row.get('plate_box_confirmed')),key=edit_key+'_boxok')
    row['plate_text']=st.text_input('Text biển nhập tay',value=row.get('plate_text',''),key=edit_key+'_text')
    layouts=['UNKNOWN','one_line','two_lines']; initial=row.get('layout','UNKNOWN')
    row['layout']=st.selectbox('Bố cục dòng',layouts,index=layouts.index(initial) if initial in layouts else 0,key=edit_key+'_layout')
    text_key=hashlib.sha256(row['plate_text'].encode()).hexdigest()[:12]
    row['text_confirmed']=st.checkbox('Tôi xác nhận text',value=bool(row.get('text_confirmed') and row.get('confirmed_text')==row['plate_text']),key=edit_key+'_textok_'+text_key)
    if row['text_confirmed']:row['confirmed_text']=row['plate_text']
    row['unreadable']=False
    def persist():store.save_draft(row,base)
    try:
        glyph_editor(row,crop,edit_key,persist,allow_proposals=False,restore_confirmations=True)
        st.write('Thứ tự hiện tại: '+''.join(c.get('label','') for c in row['characters']))
        eligible=True
        try: validate_manual(row,(image.height,image.width))
        except (ValueError,KeyError) as exc: st.info(str(exc));eligible=False
        revision={**row,'review_status':'reviewed','confirmed_by_user':True}
        if not pilot:
            if fingerprint(path)[0]!=entry['source_sha256']:raise ValueError('Ảnh nguồn đã thay đổi')
        signature=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:16]
        approved=st.checkbox('Tôi xác nhận toàn bộ box, glyph và thứ tự',key=edit_key+'_review_'+signature)
        if st.button('Lưu Glyph reviewed',disabled=not (eligible and approved and (pilot or source_ok))):
            if not pilot:
                all_completed=[r for r in latest.values() if r.get('cohort')=='final' and r.get('review_status')=='reviewed' and r['image_id']!=selected]
                validate_corpus(all_completed+[revision],store.path)
            saved=store.save(revision,base);store.clear_draft(saved['image_id']);st.rerun()
        persist()
    except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))


if __name__=='__main__':main()