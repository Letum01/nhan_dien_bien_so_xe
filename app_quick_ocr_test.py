"""Prediction-free text entry, separate from development inference UI."""
import os
from pathlib import Path

from PIL import Image
import streamlit as st

from quick_ocr_test import ROOT, STORE, TEXT_STORE, entries, key, verify_text
from final_ocr_dataset import audit_candidates, relative_image, final_reference
from annotation_store import AnnotationStore
from yolo_character_dataset import ReadOnlyStore


def main():
    st.title('Final Test — nhập text nhanh')
    st.caption('Không có dự đoán. Dùng Tab và phím mũi tên trong danh sách để chuyển ảnh; rời ô nhập để autosave.')
    if notice:=st.session_state.pop('text_saved_notice',None):
        st.success(f'Đã lưu TEXT_REVIEWED: {notice}')
    store=AnnotationStore(Path(os.environ.get('ALPR_TEXT_DB',TEXT_STORE)))
    audit=audit_candidates(write=False); items=entries(audit)
    latest=store.all_latest(); full={relative_image(r['image_path']):r for r in ReadOnlyStore(STORE).all_latest().values() if r.get('cohort')=='final'}
    st.metric('TEXT_REVIEWED riêng',sum(key(e) in latest for e in items))
    st.metric('Nhãn glyph đã duyệt được giữ',sum(full.get(relative_image(e['image']),{}).get('review_status')=='reviewed' for e in items))
    def pending(i):
        item=items[i]; previous=full.get(relative_image(item['image']),{})
        return (key(item) not in latest and item['status']!='REPLACE' and not item.get('superseded_by') and
                previous.get('review_status') not in ('reviewed','rejected'))
    if 'next_text_index' in st.session_state:
        st.session_state['text_index']=st.session_state.pop('next_text_index')
    index=st.selectbox('Ảnh trong hàng đợi',range(len(items)),format_func=lambda i:f"{i+1}. {items[i]['filename']}",
                       index=next((i for i in range(len(items)) if pending(i)),0),key='text_index')
    e=items[index]; prior=full.get(relative_image(e['image']),{})
    if e['status']=='REPLACE' or e.get('superseded_by') or prior.get('review_status')=='rejected':
        st.warning('Ứng viên đã loại hoặc thay; không gắn nhãn tại đây.');return
    with Image.open(ROOT/e['image']) as im:image=im.convert('RGB')
    w,h=image.size
    st.image(image,width=min(w,700))
    if st.checkbox('Zoom vùng biển (không lưu box)'):
        x=st.slider('Trái (%)',0,99,0); y=st.slider('Trên (%)',0,99,0)
        right=st.slider('Phải (%)',x+1,100,100); bottom=st.slider('Dưới (%)',y+1,100,100)
        st.image(image.crop((w*x//100,h*y//100,max(w*x//100+1,w*right//100),max(h*y//100+1,h*bottom//100))),width=700)
    if prior.get('review_status')=='reviewed':
        st.success('GLYPH_REVIEWED hiện hữu, chỉ đọc');st.write(prior['plate_text']);return
    ident=key(e); old=latest.get(ident,{}); base=old.get('annotation_version',0); draft=store.draft(ident)
    if draft and draft[1]!=base:st.error('Draft xung đột với revision mới.');return
    values=draft[0] if draft else old; prefix=f'{ident}:{base}'
    text=st.text_input('Text biển nhập tay',value=values.get('plate_text',''),key=prefix+'text')
    profiles=store.preference('text_source_profiles',{})
    suggested=values.get('source_group') or e.get('metadata',{}).get('source_group','')
    st.caption(f'Nhóm gợi ý từ thông tin đã nhập: {suggested or "chưa có"}. Gợi ý không xác minh nguồn.')
    source=st.selectbox('Nguồn ảnh', ['Nhập thông tin nguồn',*sorted(profiles)],key=prefix+'profile')
    remember=False
    if source=='Nhập thông tin nguồn':
        group=st.text_input('Nhóm nguồn',value=suggested,key=prefix+'group').strip()
        evidence=st.text_area('Bằng chứng nguồn và lịch sử sử dụng',value=values.get('source_evidence',''),key=prefix+'evidence').strip()
        remember=st.checkbox('Lưu nguồn này để dùng lại cho ảnh thực sự cùng nguồn',key=prefix+'remember')
    else:
        group=source; evidence=profiles[source]
        st.caption(evidence)
    import hashlib
    content=hashlib.sha256(repr((text,group,evidence)).encode()).hexdigest()
    confirmed=st.checkbox('Tôi xác nhận text đọc từ ảnh và nguồn độc lập, chưa dùng để phát triển',key=prefix+content)
    row={'image_id':ident,'image_path':e['image'],'source_sha256':e['sha256'],'source_phash':e['phash'],'split':'test',
         'plate_text':text,'source_group':group,'source_evidence':evidence,'confirmed_group':group if confirmed else None,
         'confirmed_evidence':evidence if confirmed else None,'text_confirmed':confirmed,'provenance_confirmed':confirmed,
         'confirmed_by_user':confirmed,'label_scope':'TEXT_REVIEWED','review_status':'reviewed'}
    try:
        if text or evidence:store.save_draft({**row,'review_status':'needs_review','label_scope':'TEXT_DRAFT'},base)
        if st.button('Lưu TEXT_REVIEWED',disabled=not confirmed):
            other=[final_reference(r)
                   for r in [*latest.values(),*full.values()] if r.get('review_status')=='reviewed' and r['image_id']!=ident]
            names={relative_image(item['image']):item['filename'] for item in items}
            for ref in other:ref['filename']=names.get(relative_image(ref['image']))
            verify_text(row,e,audit['known']+other)
            store.save(row,base);store.clear_draft(ident)
            if remember:store.set_preference('text_source_profiles',{**profiles,group:evidence})
            next_index=next((i for i in [*range(index+1,len(items)),*range(index)] if pending(i)),None)
            if next_index is not None:st.session_state['next_text_index']=next_index
            st.session_state['text_saved_notice']=e['filename'];st.rerun()
    except (ValueError,KeyError) as exc:st.error(str(exc))


if __name__=='__main__':main()