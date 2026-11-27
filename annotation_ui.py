"""Unified source → boxes → plate text → glyphs workflow.

Autosave covers state received by Streamlit, not unsent browser events. Local
single-user draft slot with optimistic reviewed-version conflict detection.
"""
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from annotation_store import AnnotationStore, canvas_boxes, image_id
from annotation_workflow import merge_characters, split_character, sync_characters, reorder, validate_plate_text
from dataset_validation import validate_boxes

ROOT=Path(__file__).resolve().parent


def rectangles(boxes,scale):
    return {'objects':[dict(type='rect',left=x*scale,top=y*scale,width=w*scale,height=h*scale,
                            fill='rgba(0,180,0,0.12)',stroke='green',strokeWidth=1,
                            lockRotation=True) for x,y,w,h in boxes]}


def run():
    st.title('ALPR — Gắn nhãn nhiều bước')
    store=AnnotationStore(Path(os.environ.get('ALPR_ANNOTATION_DB',str(ROOT/'data/annotations.sqlite3'))))
    @st.cache_data
    def inventory(source_root):
        return [(str(p),image_id(p)) for p in sorted(Path(source_root).rglob('*.jpg'))]
    source_root=os.environ.get('ALPR_ANNOTATION_SOURCE',str(ROOT/'dataset/license_plate_dataset'))
    sources=inventory(source_root)
    if not sources:st.warning('Không có ảnh');return
    latest=store.all_latest()
    review_mode=st.selectbox('Chế độ gắn nhãn',['Tất cả / vòng 1','Gắn nhãn vòng 2','Gắn nhãn OCR / Glyph'],
                              index=1 if os.environ.get('ALPR_ANNOTATION_ROUND')=='2' else 0)
    if review_mode=='Gắn nhãn OCR / Glyph':
        from glyph_review_ui import run as run_glyphs
        run_glyphs();return
    round2=review_mode=='Gắn nhãn vòng 2'
    if round2:
        from round2_review import load_candidates,progress
        candidates,invalid=load_candidates()
        if invalid:st.error(f'{len(invalid)} candidate lỗi; đã loại khỏi danh sách hiển thị');st.json(invalid)
        allowed={r['absolute_path'] for r in candidates}
        sources=[(p,i) for p,i in sources if str(Path(p).resolve()) in allowed]
        rank={r['absolute_path']:int(r['rank']) for r in candidates}
        sources.sort(key=lambda pair:rank[str(Path(pair[0]).resolve())])
        status_rows=progress(store,candidates)
        labels=['Tổng ứng viên','Chưa làm','Draft','Detection reviewed','Cần sửa']
        values=[len(candidates)]+[sum(r['status']==s for r in status_rows) for s in ('not_started','draft','reviewed','needs_review')]
        for col,label,value in zip(st.columns(5),labels,values):col.metric(label,value)
        import pandas as pd
        st.download_button('Tải báo cáo tiến độ vòng 2',pd.DataFrame(status_rows).to_csv(index=False).encode('utf-8-sig'),'round2_progress.csv','text/csv')
        st.download_button('Tải danh sách cần sửa',pd.DataFrame([r for r in status_rows if r['status']=='needs_review']).to_csv(index=False).encode('utf-8-sig'),'round2_needs_review.csv','text/csv')
        st.info('Vòng 2 chỉ cần detection. Text/glyph KHÔNG bắt buộc. Test pilot đóng băng; không tự gán group/split. Không chạy training.')
        train_count=sum(r['status']=='reviewed' and r['split']=='train' for r in status_rows)
        val_count=sum(r['status']=='reviewed' and r['split']=='val' for r in status_rows)
        st.caption(f'Vòng 2 reviewed: TRAIN={train_count} / VAL={val_count}. Không có thuật toán tự chia split.')
        if val_count>train_count:
            st.warning('Validation đang nhiều hơn train. Dữ liệu val không được dùng để học trọng số. Kiểm tra split của từng nguồn; không đổi reviewed tự động.')
        auto_next=st.checkbox('Sang ảnh tiếp theo sau khi xác nhận Detection reviewed',value=True)
        if not sources:st.warning('Không có candidate hợp lệ trong nguồn');return
    reviewed=sum(latest.get(i,{}).get('review_status')=='reviewed' for _,i in sources)
    needs=sum(latest.get(i,{}).get('review_status')=='needs_review' for _,i in sources)
    unlabeled=sum(i not in latest or latest[i].get('review_status')=='unlabeled' for _,i in sources)
    if not round2:
        columns=st.columns(4)
        for col,label,value in zip(columns,['Tổng ảnh','Reviewed detection','Chưa gắn nhãn','Needs review'],[len(sources),reviewed,unlabeled,needs]):col.metric(label,value)
    st.progress(reviewed/len(sources))
    st.caption('Reviewed detection không có nghĩa đủ điều kiện OCR/train. Còn cần text/glyph, split và validator.')
    round_path=ROOT/'reports/annotation_round_01.json'
    round_only=st.checkbox('Chỉ hàng đợi vòng 1 (150 ảnh)',value=False if round2 else os.environ.get('ALPR_ROUND_ONLY')=='1',disabled=round2)
    if round2:round_only=False
    queue_paths=None
    if round_path.exists():
        queue=json.loads(round_path.read_text(encoding='utf-8'))['images']
        queue_paths={str((ROOT/r['image_path']).resolve()) for r in queue}
        queue_rows=[latest.get(i,{}) for p,i in sources if str(Path(p).resolve()) in queue_paths]
        queued_reviewed=sum(r.get('review_status')=='reviewed' for r in queue_rows)
        st.caption(f"Vòng 1: {len(queue_rows)} ảnh; reviewed {queued_reviewed}; chờ duyệt {len(queue_rows)-queued_reviewed}; "
                   f"train/val/test reviewed: {[sum(r.get('review_status')=='reviewed' and r.get('split')==s for r in queue_rows) for s in ('train','val','test')]}. Validator toàn dataset chưa chạy.")
    query=st.text_input('Tìm tên ảnh')
    status_filter=st.selectbox('Lọc trạng thái',['all','unlabeled','reviewed','needs_review','proposed','rejected'])
    filtered=[(p,i) for p,i in sources if (not round_only or queue_paths is None or str(Path(p).resolve()) in queue_paths) and query.lower() in Path(p).name.lower() and
              (status_filter=='all' or latest.get(i,{}).get('review_status','unlabeled')==status_filter)]
    if not filtered:st.info('Không có ảnh phù hợp');return
    paths=[p for p,_ in filtered]
    if round_only and round_path.exists():
        ranks={str((ROOT/r['image_path']).resolve()):r['rank'] for r in queue}
        paths.sort(key=lambda p:ranks.get(str(Path(p).resolve()),99999))
    state=st.session_state
    if state.get('pending_review_next'):
        state['selected_source']=state.pop('pending_review_next')

    def persist_active():
        if state.get('working_annotation'):
            store.save_draft(state['working_annotation'],state['base_version'])

    def navigate(target):
        try:
            persist_active()
        except ValueError as exc:
            state['navigation_error']=str(exc)
            state['selected_source']=state['active_source']
            return
        state['selected_source']=target
        store.set_preference('last_source',target)

    if 'selected_source' not in state:
        remembered=store.preference('last_source',paths[0])
        state['selected_source']=remembered if remembered in paths else paths[0]
    if state['selected_source'] not in paths:
        navigate(paths[0])
    current=state['selected_source']
    nav=st.columns(2)
    index=paths.index(current)
    nav[0].button('← Ảnh trước',disabled=index==0,on_click=navigate,args=(paths[max(0,index-1)],))
    nav[1].button('Ảnh tiếp theo →',disabled=index==len(paths)-1,on_click=navigate,args=(paths[min(len(paths)-1,index+1)],))
    def select_changed():navigate(state['selected_source'])
    st.selectbox('Ảnh nguồn',paths,key='selected_source',on_change=select_changed)
    current=state['selected_source']
    if state.get('navigation_error'):
        st.error(state.pop('navigation_error'));return
    if state.get('active_source')!=current:
        identity=dict(sources)[current]
        previous=store.latest(identity) or {}
        draft=store.draft(identity)
        base=previous.get('annotation_version',0)
        if draft and draft[1]!=base:
            st.error('Draft cũ xung đột với reviewed revision mới; cần giải quyết ở lịch sử. Không ghi đè.');return
        state['working_annotation']=draft[0] if draft else {
            **previous,'image_id':identity,'image_path':os.path.relpath(current,ROOT),
            'boxes':previous.get('boxes',[]),'plates':previous.get('plates',[]),
            'source_group':previous.get('source_group',''),'split':previous.get('split','UNASSIGNED'),
            'review_status':previous.get('review_status','unlabeled')}
        state['base_version']=base;state['active_source']=current
        state['canvas_epoch']=state.get('canvas_epoch',0)+1
        store.set_preference('last_source',current)
    row=state['working_annotation'];identity=row['image_id']
    st.caption(f'{index+1}/{len(paths)} — autosave draft sau mỗi cập nhật server; không đóng trang khi canvas chưa gửi thay đổi.')
    image=Image.open(current).convert('RGB')
    row['source_group']=st.text_input('source_group',value=row.get('source_group',''),key=f'group_{identity}')
    group_splits=sorted({old.get('split') for old in latest.values()
                         if old.get('source_group')==row['source_group'] and row['source_group'] and old.get('review_status')=='reviewed'})
    if group_splits:
        st.info(f"source_group {row['source_group']!r} đã có reviewed split: {', '.join(group_splits)}. Không tự chọn thay bạn.")
    split_options=['UNASSIGNED','train','val'] if round2 else ['UNASSIGNED','train','val','test']
    if row.get('split') not in split_options:
        st.error('Split hiện lưu không phù hợp chế độ vòng 2; không tự sửa. Kiểm tra nhãn trong chế độ tất cả.');return
    row['split']=st.selectbox('Split',split_options,index=split_options.index(row.get('split','UNASSIGNED')),key=f'split_{identity}')
    if group_splits and row['split'] not in group_splits:
        st.error('Split đang chọn khác split của source_group đã reviewed; lưu sẽ bị từ chối.')
    with st.expander('Ảnh gần trùng — chỉ hỗ trợ duyệt, không sao chép nhãn'):
        report=ROOT/'reports/dataset_audit.json'
        if report.exists():
            audit=json.loads(report.read_text(encoding='utf-8'))
            relative=row['image_path']
            pairs=[pair for pair in audit.get('near_duplicate_candidates',[]) if relative in (pair['a'],pair['b'])][:6]
            for pair in pairs:
                other=pair['b'] if pair['a']==relative else pair['a']
                st.image(str(ROOT/other),caption=other,width=260)
    tabs=st.tabs(['1. Bounding boxes','2. Văn bản biển số','3. Ký tự','4. Lịch sử / xuất'])
    with tabs[0]:
        index_path=ROOT/'reports/round_01_proposals/index.json'
        if index_path.exists():
            proposal_report=json.loads(index_path.read_text(encoding='utf-8'))
            suggestion=next((r for r in proposal_report['results'] if r['image_id']==identity),None)
            if suggestion and suggestion.get('proposals'):
                with st.expander('Gợi ý chuẩn bị sẵn — màu vàng, chưa duyệt',expanded=True):
                    st.image(str(ROOT/suggestion['preview']),use_column_width=True)
                    choices=st.multiselect('Chọn proposal để sửa (không tự chọn)',
                                           range(len(suggestion['proposals'])),key=f'proposals_{identity}')
                    if st.button('Đưa proposal đã chọn vào canvas'):
                        if not choices:
                            st.warning('Chọn proposal hoặc tự vẽ box. Không có proposal được duyệt tự động.')
                        else:
                            row['boxes']=[suggestion['proposals'][i]['bbox'] for i in choices]
                            row['plates']=[];row['review_status']='proposed';row['confirmed_by_user']=False
                            state['canvas_epoch']+=1;persist_active();st.rerun()
        scale=st.slider('Zoom',.25,2.0,1.0,.25)
        mode=st.radio('Canvas mode',['rect','transform'],horizontal=True)
        st.caption('rect vẽ; transform kéo/resize; toolbar undo/delete. Không xoay rectangle.')
        if row['boxes']:
            st.caption('Box hiện lưu (pixel gốc): '+json.dumps(row['boxes']))
            remove_indices=st.multiselect('Xóa box theo index (kể cả box quá nhỏ)',
                                          range(len(row['boxes'])),
                                          key=f"remove_boxes_{identity}_{state['canvas_epoch']}")
            if st.button('Xóa chính xác các box đã chọn',disabled=not remove_indices):
                row['boxes']=[box for index,box in enumerate(row['boxes']) if index not in remove_indices]
                row['plates']=[]
                row['review_status']='needs_review'
                row['confirmed_by_user']=False
                state['canvas_epoch']+=1
                persist_active();st.rerun()
        if st.button('Nạp gợi ý OpenCV — proposed'):
            from localization import locate_license_plates
            row['boxes']=[list(box) for _,box,_ in locate_license_plates(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))]
            row['plates']=[];row['review_status']='proposed';state['canvas_epoch']+=1
        if st.button('Xóa tất cả box / từ chối proposal'):
            row['boxes']=[];row['plates']=[];row['review_status']='rejected';state['canvas_epoch']+=1
            persist_active();st.rerun()
        canvas=st_canvas(background_image=image,width=int(image.width*scale),height=int(image.height*scale),
                         initial_drawing=rectangles(row['boxes'],scale),drawing_mode=mode,
                         stroke_color='green',stroke_width=1,update_streamlit=True,
                         key=f"source_canvas_{identity}_{scale}_{state['canvas_epoch']}")
        if canvas.json_data is not None:
            try:
                boxes=canvas_boxes(canvas.json_data.get('objects',[]),scale,image.width,image.height)
                validate_boxes(boxes,image.width,image.height)
                ready_key=f"source_ready_{identity}_{scale}_{state['canvas_epoch']}"
                if boxes:state[ready_key]=True
                initialized=bool(boxes) or not row['boxes'] or state.get(ready_key,False)
                if initialized and boxes!=row['boxes']:
                    row['boxes']=boxes;row['plates']=[];row['review_status']='needs_review'
                    row['confirmed_by_user']=False
            except ValueError as exc:
                st.error(str(exc))
                st.info('Sửa/xóa box lỗi trên canvas hoặc dùng danh sách index ở phía trên. Không thể duyệt trước khi box hợp lệ.')
                return
        # Approval widget changes identity whenever geometry changes, so a stale
        # checkmark cannot approve newly edited boxes on the next click.
        geometry_key=__import__('hashlib').sha256(json.dumps(row['boxes']).encode()).hexdigest()[:12]
        confirmed=st.checkbox('Tôi đã kiểm tra tất cả box; [] xác nhận ảnh không có biển',key=f"approve_{identity}_{geometry_key}")
        split_confirmation=True
        if round2:
            split_confirmation=st.checkbox(f"Tôi xác nhận nguồn {row['source_group'] or '(chưa nhập)'} thuộc split {row['split']}; không chỉ là giá trị hiển thị",
                                           key=f"split_ack_{identity}_{row['source_group']}_{row['split']}")
        if st.button('Xác nhận detection reviewed',disabled=not confirmed or not split_confirmation):
            try:
                candidate={**row,'review_status':'reviewed','confirmed_by_user':True,'human_reviewed':True}
                from label_detection import validate_annotation
                validate_annotation(candidate)
                if round2:
                    from round2_review import enforce_plan
                    enforce_plan(candidate,latest)
                if any(old.get('source_group')==row['source_group'] and old.get('split')!=row['split'] for old in latest.values()):
                    raise ValueError('source_group đã dùng trong split khác')
                saved=store.save(candidate,state['base_version'])
                state['base_version']=saved['annotation_version'];row.update(saved)
                from prepare_annotation_round import readiness
                if round_path.exists():
                    status=readiness(store,json.loads(round_path.read_text(encoding='utf-8'))['images'])
                    store.set_preference('last_validation',status)
                st.success('Đã lưu reviewed revision; draft chỉnh sửa sau đó không tự thay nhãn này')
                if round2:
                    persist_active()
                    if auto_next and index+1<len(paths):
                        state['pending_review_next']=paths[index+1]
                        store.set_preference('last_source',paths[index+1])
                    st.rerun()
            except ValueError as exc:st.error(str(exc))
    # Geometry changes clear downstream text/glyph review, never transfer labels by index.
    while len(row['plates'])<len(row['boxes']):
        row['plates'].append({'plate_text':'','unreadable':False,'layout':'UNKNOWN','text_confirmed':False,'characters':[]})
    with tabs[1]:
        for pi,(box,plate) in enumerate(zip(row['boxes'],row['plates'])):
            x,y,w,h=box;st.image(image.crop((x,y,x+w,y+h)),caption=f'Biển {pi}',width=350)
            box_key=__import__('hashlib').sha256(json.dumps(box).encode()).hexdigest()[:12]
            plate['plate_text']=st.text_input('Text gốc (không tự sửa)',value=plate['plate_text'],key=f'text_{identity}_{box_key}_{pi}')
            plate['unreadable']=st.checkbox('Không đọc được / thiếu ký tự',value=plate['unreadable'],key=f'unreadable_{identity}_{pi}')
            plate['layout']=st.selectbox('Số dòng',['UNKNOWN','one_line','two_lines'],index=['UNKNOWN','one_line','two_lines'].index(plate['layout']),key=f'layout_{identity}_{pi}')
            text_key=__import__('hashlib').sha256((plate['plate_text']+str(plate['unreadable'])+box_key).encode()).hexdigest()[:12]
            plate['text_confirmed']=st.checkbox('Tôi xác nhận text đúng với crop',key=f'text_confirm_{identity}_{pi}_{text_key}')
    with tabs[2]:
        if row['boxes']:
            pi=st.selectbox('Biển cần gắn glyph',range(len(row['boxes'])))
            plate=row['plates'][pi];x,y,w,h=row['boxes'][pi]
            crop=image.crop((x,y,x+w,y+h))
            editor_key=f'open_glyph_{identity}_{pi}_{state["canvas_epoch"]}'
            if st.button('Mở canvas ký tự',key='button_'+editor_key):
                state[editor_key]=True
            if state.get(editor_key):
                glyph_editor(plate,crop,f'{identity}_{pi}_{state["canvas_epoch"]}',persist_active)
            else:
                st.info('Mở canvas khi tab đang hiển thị để component khởi tạo đúng kích thước.')
        else:st.info('Vẽ box trước')
    with tabs[3]:
        if st.button('Kiểm tra nhãn reviewed và điều kiện export'):
            from prepare_annotation_round import readiness
            status=readiness(store,queue if round_path.exists() else [])
            store.set_preference('last_validation',status)
        last_validation=store.preference('last_validation')
        if last_validation:
            st.json(last_validation)
        st.caption('Thay đổi nội dung sẽ yêu cầu xác nhận lại. Draft không thay reviewed revision; xuất chỉ lấy bản reviewed mới nhất.')
        content_key=__import__('hashlib').sha256(json.dumps([row['boxes'],row['plates'],row['source_group'],row['split']],sort_keys=True).encode()).hexdigest()[:12]
        final_confirm=st.checkbox('Tôi xác nhận toàn bộ box/text/glyph đã nhập; lưu reviewed revision mới',key=f'final_{content_key}')
        if st.button('Lưu toàn bộ nhãn reviewed',disabled=not final_confirm):
            try:
                candidate={**row,'review_status':'reviewed','confirmed_by_user':True,'human_reviewed':True}
                from label_detection import validate_annotation
                validate_annotation(candidate)
                if round2:
                    from round2_review import enforce_plan
                    enforce_plan(candidate,latest)
                for plate in candidate['plates']:
                    validate_plate_text(plate)
                if any(old.get('source_group')==row['source_group'] and old.get('split')!=row['split'] for old in latest.values()):
                    raise ValueError('source_group đã được gán split khác')
                saved=store.save(candidate,state['base_version']);row.update(saved)
                state['base_version']=saved['annotation_version']
                from prepare_annotation_round import readiness
                store.set_preference('last_validation',readiness(store,queue if round_path.exists() else []))
                st.success('Đã lưu revision đầy đủ; có thể xuất snapshot')
            except ValueError as exc:st.error(str(exc))
        history=store.history(identity)
        st.json(history)
        if history:
            version=st.selectbox('Revision phục hồi',[r['annotation_version'] for r in history])
            if st.button('Khôi phục thành needs_review revision mới'):
                try:
                    saved=store.restore(identity,version,state['base_version']);store.clear_draft(identity)
                    state.pop('active_source',None);st.rerun()
                except ValueError as exc:st.error(str(exc))
        destination=st.text_input('Thư mục export MỚI',value='datasets/annotation_export')
        if st.button('Xuất snapshot reviewed từ SQLite'):
            try:
                if round2:
                    from round2_review import check_frozen_pilot
                    check_frozen_pilot(list(store.all_latest().values()))
                export_snapshot(store,ROOT/destination);st.success('Đã xuất; chạy validator trước training')
            except (ValueError,FileExistsError) as exc:st.error(str(exc))
    try:persist_active()
    except ValueError as exc:st.error(str(exc))


def glyph_editor(plate,crop,key,persist=lambda: None, *, allow_proposals=True, restore_confirmations=False):
    from streamlit_drawable_canvas import st_canvas
    chars=plate['characters']
    epoch_key='glyph_epoch_'+key
    st.session_state.setdefault(epoch_key,0)
    if allow_proposals and st.button('Tạo glyph proposal chưa nhãn',key='proposal_'+key):
        from segmentation import segment_characters
        chars[:]=[dict(bbox=list(b),label='',line=0,order=i,confirmed=False) for i,(_,b) in enumerate(segment_characters(cv2.cvtColor(np.array(crop),cv2.COLOR_RGB2BGR)))]
        st.session_state[epoch_key]+=1
        persist();st.rerun()
    zoom=st.slider('Glyph zoom',1.0,4.0,2.0,.5,key='zoom_'+key)
    mode=st.radio('Glyph canvas',['rect','transform'],key='mode_'+key,horizontal=True)
    canvas=st_canvas(background_image=crop,width=int(crop.width*zoom),height=int(crop.height*zoom),
                     initial_drawing=rectangles([c['bbox'] for c in chars],zoom),drawing_mode=mode,
                     stroke_width=1,stroke_color='green',key=f'canvas_glyph_{key}_{zoom}_{st.session_state[epoch_key]}')
    if canvas.json_data is not None:
        boxes=canvas_boxes(canvas.json_data.get('objects',[]),zoom,crop.width,crop.height)
        ready_key=f'glyph_ready_{key}_{zoom}_{st.session_state[epoch_key]}'
        if boxes:
            st.session_state[ready_key]=True
        # Custom component emits an empty initialization value before Fabric
        # loads initial_drawing. That is NOT a user deleting restored glyphs.
        if boxes or not chars or st.session_state.get(ready_key):
            chars[:]=sync_characters(chars,boxes)
    # Split/merge actions return before the rest of the render and recreate
    # the component with a new epoch. Never reapply an old widget selection.
    if chars:
        indices=st.multiselect('Chọn index glyph',range(len(chars)),key=f'selection_{key}_{st.session_state[epoch_key]}')
        explicit_indices=st.text_input('Index thao tác (tùy chọn, ví dụ 0,1)',key=f'indices_text_{key}_{st.session_state[epoch_key]}')
        if explicit_indices.strip():
            try:
                indices=sorted(set(int(i.strip()) for i in explicit_indices.split(',')))
                if any(i<0 or i>=len(chars) for i in indices):raise ValueError('Index ngoài danh sách glyph')
            except ValueError as exc:st.error(str(exc));return
        if st.button('Xóa glyph đã chọn',key='delete_'+key):
            plate['characters']=[c for i,c in enumerate(chars) if i not in indices]
            st.session_state[epoch_key]+=1;persist();st.rerun()
        offset=st.number_input('Split offset pixel trong glyph',min_value=1,value=5,key='offset_'+key)
        if st.button('Split glyph',key='split_'+key):
            try:
                if len(indices)!=1:raise ValueError('Chọn đúng một glyph')
                plate['characters']=split_character(chars,indices[0],int(offset));st.session_state[epoch_key]+=1;persist();st.rerun()
            except ValueError as exc:st.error(str(exc))
        if st.button('Merge glyph cùng dòng',key='merge_'+key):
            try:
                plate['characters']=merge_characters(chars,indices);st.session_state[epoch_key]+=1;persist();st.rerun()
            except ValueError as exc:st.error(str(exc))
        for index,char in enumerate(chars):
            x,y,w,h=char['bbox'];st.image(crop.crop((x,y,x+w,y+h)),caption=f'#{index}',width=90)
            version=st.session_state[epoch_key]
            char['label']=st.text_input('Nhãn glyph',value=char.get('label',''),key=f'label_{key}_{version}_{index}',max_chars=1)
            char['line']=st.number_input('Dòng (0=trên)',min_value=0,max_value=1,value=int(char.get('line',0)),key=f'line_{key}_{version}_{index}')
            glyph_key=__import__('hashlib').sha256(json.dumps([char['bbox'],char['label'],char['line']]).encode()).hexdigest()[:12]
            char['confirmed']=st.checkbox('Đã kiểm tra glyph đủ nét',key=f'confirm_{key}_{version}_{index}_{glyph_key}',
                                          value=bool(restore_confirmations and char.get('confirmed') and char.get('confirmation_key')==glyph_key))
            if restore_confirmations:
                char['confirmation_key']=glyph_key if char['confirmed'] else None
        if st.button('Sắp xếp dòng / trái → phải',key='reorder_'+key):
            plate['characters']=reorder(chars);st.session_state[epoch_key]+=1;persist();st.rerun()


def export_snapshot(store,destination):
    """NEW snapshot only, latest reviewed revisions not autosaved drafts."""
    rows=[r for r in store.all_latest().values() if r.get('review_status')=='reviewed']
    if not rows:raise ValueError('Chưa có reviewed annotation để xuất')
    from label_detection import export_yolo, validate_annotation
    for row in rows:validate_annotation({**row,'human_reviewed':True})
    from dataset_validation import check_split_leakage
    if {row['split'] for row in rows}!={'train','val','test'}:
        raise ValueError('Cần reviewed train/val/test trước khi xuất dataset chính thức')
    check_split_leakage(rows,ROOT)
    for row in rows:
        for plate in row.get('plates',[]):
            if plate.get('text_confirmed'):
                validate_plate_text(plate)
                for char in plate.get('characters',[]):
                    if char.get('confirmed') and char.get('label') not in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        raise ValueError('Invalid reviewed glyph label')
    destination.mkdir(parents=True,exist_ok=False)
    manifest=destination/'reviewed_detection.json'
    manifest.write_text(json.dumps([{**row,'human_reviewed':True} for row in rows],ensure_ascii=False,indent=2),encoding='utf-8')
    # OCR exported separately only if the SAME reviewed revision includes text.
    from label_crops import save_label
    from label_characters import export_reviewed_glyphs
    for row in rows:
        source=cv2.imread(str(ROOT/row['image_path']))
        for pi,(box,plate) in enumerate(zip(row['boxes'],row.get('plates',[]))):
            try:validate_plate_text(plate)
            except ValueError:continue
            x,y,w,h=box;folder=destination/'crops';folder.mkdir(exist_ok=True)
            path=folder/f"{row['image_id'][:16]}_{pi}.png";cv2.imwrite(str(path),source[y:y+h,x:x+w])
            save_label(destination/'reviewed_crops.csv',{'image_path':str(path.relative_to(destination)),
                       'plate_text':plate['plate_text'],'annotation_status':'human_reviewed',
                       'source_group':row['source_group'],'split':'validation' if row['split']=='val' else row['split'],
                       'image_id':row['image_id'],'line_layout':plate['layout']})
            if plate['characters'] and all(c.get('confirmed') for c in plate['characters']):
                export_reviewed_glyphs(path,plate['characters'],destination/'glyphs'/path.stem,row['source_group'],
                                      'validation' if row['split']=='val' else row['split'])
    export_yolo(manifest,destination/'license_plates')