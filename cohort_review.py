"""Manual annotation QC for the fixed progressive cohort, isolated from training labels."""
import csv
import io
import os
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import streamlit as st

from external_dataset_prepare import AUDIT, ROOT, crop, digest, draw, load_image, safe_path
from final_ocr_dataset import read

QUEUE=ROOT/'reports/progressive_readiness_review_v1'
CURATION=ROOT/'reports/task_curation_20261127T081828802935Z'
EXTRACTED=ROOT/'external_datasets/extracted_20261127T053300134736Z/OCR_Data'
INPUTS={
    'glyph':('glyph_review_inputs.json','ae909323e855c1a6e1086a286f7fb8e5c4fccd78145c558f8ba5dd7ca3256ef8',412),
    'detector':('detector_review_inputs.json','55d985d7ec1596c6b16eb85d404afdbeb33006bd9a5f8bacacb897312df84659',122),
}
DECISIONS={'glyph':{'correct':'Đúng','incorrect':'Sai','unclear':'Không rõ'},
           'detector':{'complete':'Đủ box','incomplete':'Thiếu box','unclear':'Không rõ'}}
DOMAINS={'unknown':'Chưa xác định','vn_appearance':'Có hình thức biển Việt Nam','foreign':'Biển nước ngoài',
         'chart':'Bảng mẫu ký tự','synthetic':'Ảnh tổng hợp (có bằng chứng)','other':'Khác'}
ERRORS={'label':'Sai nhãn ký tự','clipped':'Box cắt mất nét','extra':'Box thừa','missing':'Thiếu box',
        'overlap':'Box chồng lấn','multiple':'Một box chứa nhiều ký tự','poor_pixels':'Ảnh khó đọc','other':'Khác'}
RESEARCH_DOMAINS={'vn_appearance':'Vietnamese-looking unverified','foreign':'Foreign plate','chart':'Character chart'}
BUCKETS=('Vietnamese verified','Vietnamese-looking unverified','Foreign plate','Character chart','Ambiguous','Invalid')
FOCUS=set('DTMNKABS')


@st.cache_data
def inputs(kind, pinned_hash):
    name, expected, size=INPUTS[kind]
    if pinned_hash!=expected: raise ValueError('Cohort input hash changed')
    rows=read(QUEUE/name)
    keys={(r['parent'],r.get('glyph_index')) for r in rows}
    if len(rows)!=size or len(keys)!=size: raise ValueError('Cohort input count or IDs changed')
    return rows


@st.cache_data
def metadata():
    return {r['path']:r for r in read(AUDIT/'OCR_Data_manifest.json')}


@st.cache_data
def priority_data():
    wanted={(r['parent'],r['glyph_index']) for r in read(QUEUE/INPUTS['glyph'][0])}
    quality={(r['parent'],r['glyph_index']):r['tier'] for r in read(CURATION/'glyphs.json')
             if (r['parent'],r['glyph_index']) in wanted}
    counts=Counter(r['class_label'] for r in read(QUEUE/'baseline370.json')['rows'])
    paths=read(CURATION/'task_review_index.json')
    names={'Vietnamese plate appearance':'vn_appearance','Foreign plate glyph':'foreign','Character chart':'chart'}
    hints={paths[i-1]:names[o['domain_candidate']] for o in read(CURATION/'visual_review.json')['observations']
           if o.get('domain_candidate') in names for i in o['tiles']}
    return quality,counts,hints


def research_bucket(row, decision, quality, hints):
    if quality.get((row['parent'],row['glyph_index']))=='D' or decision.get('decision')=='incorrect': return 'Invalid'
    if decision.get('decision')=='unclear': return 'Ambiguous'
    domain=decision.get('domain',hints.get(row['parent']))
    # This store records visual observations, never verified Vietnamese provenance.
    return RESEARCH_DOMAINS.get(domain,'Ambiguous')


def glyph_order(rows, latest, quality, counts, hints):
    """Sort pending IDs, not the frozen rows. Keep siblings contiguous."""
    observations=defaultdict(set)
    for i,r in latest.items():
        if r['domain'] in RESEARCH_DOMAINS: observations[rows[i-1]['parent']].add(r['domain'])
    def domain_rank(parent):
        domains=observations.get(parent)
        value=next(iter(domains)) if domains and len(domains)==1 else hints.get(parent) if not domains else None
        return 0 if value=='vn_appearance' else 2 if value in ('foreign','chart') else 1
    def rank(i):
        r=rows[i-1]; c=r['character']; n=counts[c]
        return (c not in FOCUS and n>2,quality.get((r['parent'],r['glyph_index']))!='B',c not in FOCUS,n,i)
    parents=defaultdict(list)
    for i,row in enumerate(rows,1):
        if i not in latest: parents[row['parent']].append(i)
    return [i for p,ids in sorted(parents.items(),key=lambda v:(domain_rank(v[0]),min(rank(i)[:-1] for i in v[1]),-len(v[1]),min(v[1])))
            for i in sorted(ids,key=rank)]


def source_image(entry, audit):
    path=safe_path(EXTRACTED,entry['parent'])
    label=safe_path(EXTRACTED,str(Path(entry['parent']).parent.parent.parent/'labels'/path.parent.name/path.with_suffix('.txt').name))
    if (audit.get('annotation')!='ok' or digest(path)!=entry['sha256'] or entry['sha256']!=audit['sha256']
            or digest(label)!=audit['label_sha256']):
        raise ValueError('Ảnh hoặc nhãn nguồn không khớp audit; không thể lưu mẫu này.')
    im=load_image(path)
    if im.shape[:2]!=(audit['height'],audit['width']): raise ValueError('Kích thước ảnh không khớp audit.')
    return im


class Reviews:
    def __init__(self,path):
        self.path=Path(path)

    def latest(self, kind):
        if not self.path.is_file(): return {}
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory=sqlite3.Row
            return {r['review_id']:dict(r) for r in db.execute('''SELECT r.* FROM reviews r JOIN
                (SELECT review_id,MAX(revision) AS newest FROM reviews WHERE cohort=? GROUP BY review_id) n
                ON r.review_id=n.review_id AND r.revision=n.newest WHERE r.cohort=?''',(INPUTS[kind][1],)*2)}

    def save(self,kind,idx,revision,decision,domain,error,note,reviewer):
        self.save_many(kind,[(idx,revision,decision,domain,error,note)],reviewer)

    def save_many(self,kind,items,reviewer):
        reviewer=reviewer.strip()
        if not reviewer or len(reviewer)>60: raise ValueError('Nhập bí danh người rà soát (tối đa 60 ký tự).')
        if not items or len({r[0] for r in items})!=len(items): raise ValueError('Chọn quyết định riêng cho ít nhất một mẫu.')
        for idx,revision,decision,domain,error,note in items:
            if not 1<=idx<=INPUTS[kind][2] or decision not in DECISIONS[kind] or domain not in DOMAINS or error not in {*ERRORS,''}:
                raise ValueError('Lựa chọn không hợp lệ.')
            if len(note.strip())>500: raise ValueError('Ghi chú tối đa 500 ký tự.')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path,timeout=10)) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS reviews(cohort TEXT, review_id INTEGER, revision INTEGER,
                decision TEXT, domain TEXT, error_type TEXT, reason TEXT, reviewer TEXT,
                saved_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(cohort,review_id,revision))''')
            db.execute('BEGIN IMMEDIATE')
            cohort=INPUTS[kind][1]
            for idx,revision,decision,domain,error,note in items:
                current=db.execute('SELECT COALESCE(MAX(revision),0) FROM reviews WHERE cohort=? AND review_id=?',(cohort,idx)).fetchone()[0]
                if revision!=current: raise ValueError('Mẫu vừa được cập nhật ở phiên khác; tải lại trước khi lưu.')
                db.execute('INSERT INTO reviews(cohort,review_id,revision,decision,domain,error_type,reason,reviewer) VALUES(?,?,?,?,?,?,?,?)',
                           (cohort,idx,current+1,decision,domain,error,note.strip(),reviewer))


def statistics(kind, rows, latest):
    groups=defaultdict(Counter); domains=defaultdict(Counter); errors=Counter(); strata=defaultdict(Counter)
    for idx,row in enumerate(rows,1):
        r=latest.get(idx); outcome=r['decision'] if r else 'pending'
        if kind=='glyph': groups[row['character']][outcome]+=1
        for s in row['strata']: strata[s][outcome]+=1
        if r:
            domains[DOMAINS[r['domain']]][outcome]+=1
            if r['error_type']: errors[ERRORS[r['error_type']]]+=1
    return groups,domains,errors,strata


def exported(rows, latest, buckets=None):
    fields=['review_id','decision','reason','reviewer','domain','research_domain','error_type','revision','saved_at','parent','sha256','glyph_index','character']
    buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=fields); writer.writeheader()
    def safe(value):
        return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')) else value
    for idx,row in enumerate(rows,1):
        content={**row,**latest.get(idx,{}),'review_id':idx}
        if buckets is not None: content['research_domain']=buckets[idx]
        writer.writerow({f:safe(content.get(f,'')) for f in fields})
    return buffer.getvalue().encode('utf-8-sig')


def grouped_review(im, ids, rows, latest, key, db, reviewer, pending):
    st.caption('Chọn riêng từng glyph. Ô chưa chọn không được lưu. Không cập nhật lại glyph đã duyệt trong chế độ này.')
    overlay=draw(im.copy(),[[str(i),*rows[i-1]['pixel_xywh']] for i in ids],(0,220,0))
    st.image(overlay,channels='BGR',use_column_width=True,caption='Mã mẫu trên ảnh nguồn')
    if st.checkbox('Xem ảnh gốc không khung'): st.image(im,channels='BGR',use_column_width=True)
    items=[]
    with st.form('group_'+str(ids[0])):
        for i in ids:
            row=rows[i-1]; roi,_=crop(im,row['pixel_xywh'])
            left,right=st.columns([1,3]); left.image(roi,channels='BGR',width=96)
            left.caption(f"#{i} · Nhãn nguồn {row['character']}")
            if i in latest:
                right.write('Đã lưu: '+DECISIONS['glyph'][latest[i]['decision']]); continue
            pick=right.selectbox(f'Quyết định #{i}',['',*DECISIONS['glyph']],format_func=lambda d:DECISIONS['glyph'].get(d,'Chưa chọn'))
            domain=right.selectbox(f'Domain #{i}',list(DOMAINS),format_func=DOMAINS.get)
            error=right.selectbox(f'Loại lỗi #{i}',['',*ERRORS],format_func=lambda e:ERRORS.get(e,'Không ghi loại lỗi'))
            if pick: items.append((i,0,pick,domain,error,''))
        submit=st.form_submit_button('Lưu các quyết định đã chọn')
    if submit:
        try:
            db.save_many('glyph',items,reviewer)
            done={r[0] for r in items}
            remaining=[i for i in pending if i not in done]
            target=next((i for i in ids if i in remaining), remaining[0] if remaining else ids[0])
            st.session_state['cohort_saved']=f'Đã lưu {len(done)} quyết định riêng.'
            st.session_state['cohort_next']=(key,target); st.rerun()
        except (ValueError,sqlite3.Error,OSError) as exc: st.error(str(exc))


def render():
    st.header('Kiểm định cohort OCR')
    st.caption('Chỉ đọc ảnh và annotation có sẵn. Không chạy model, không sửa box, không xác nhận quyền train.')
    kind=st.radio('Hàng đợi',('glyph','detector'),format_func=lambda x:'Glyph (412)' if x=='glyph' else 'Độ đầy đủ box (122)',horizontal=True)
    db=Reviews(os.environ.get('ALPR_COHORT_REVIEW_DB',ROOT/'data/alpr_cohort_review.sqlite3'))
    try:
        queues={k:inputs(k,digest(QUEUE/spec[0])) for k,spec in INPUTS.items()}
        progress={k:db.latest(k) for k in INPUTS}
    except (OSError,ValueError,sqlite3.Error) as exc:
        st.error(str(exc)); return
    st.write(f"Glyph: {len(progress['glyph'])}/412 · Độ đầy đủ box: {len(progress['detector'])}/122")
    rows,latest=queues[kind],progress[kind]
    grouped=False; buckets=None; order='goc'
    if kind=='glyph':
        order=st.radio('Thứ tự hiển thị',('uutien','goc'),format_func=lambda k:'Ưu tiên' if k=='uutien' else 'Mã mẫu gốc',horizontal=True)
        grouped=st.checkbox('Review nhiều glyph cùng ảnh')
        try:
            quality, counts, hints=priority_data()
            buckets={i:research_bucket(row,latest.get(i,{}),quality,hints) for i,row in enumerate(rows,1)}
        except (OSError,KeyError,ValueError) as exc:
            st.error(f'Không tải được metadata ưu tiên: {exc}'); return
    st.progress(len(latest)/len(rows))
    st.caption('Các stratum giao nhau và có chủ đích lấy dư mẫu khó; không suy rộng tỷ lệ lỗi của cohort cho toàn dataset.')
    with st.expander('Thống kê và xuất kết quả'):
        outcomes=Counter(r['decision'] for r in latest.values())
        bad,good=('incorrect','correct') if kind=='glyph' else ('incomplete','complete')
        resolved=outcomes[bad]+outcomes[good]
        rate=f'{outcomes[bad]}/{resolved} ({outcomes[bad]/resolved:.1%})' if resolved else 'chưa có kết luận'
        st.write(f"{'Sai nhãn/crop' if kind=='glyph' else 'Thiếu box'} trên mẫu đã kết luận: {rate}. Không rõ: {outcomes['unclear']}.")
        for title, values in zip(('Theo class nguồn','Theo domain được người review ghi','Theo loại lỗi ghi nhận','Theo stratum lấy mẫu'),statistics(kind,rows,latest)):
            if values:
                st.markdown(f'**{title}**')
                st.dataframe([{'nhóm':k,**v} if isinstance(v,Counter) else {'loại':k,'số mẫu':v} for k,v in sorted(values.items())],hide_index=True)
        if buckets:
            tally=Counter(buckets.values()); st.markdown('**Nhóm nghiên cứu (không phải quyền hoặc provenance)**')
            st.dataframe([{'nhóm':b,'số glyph':tally[b]} for b in BUCKETS],hide_index=True)
        st.download_button('Tải CSV tiến độ',exported(rows,latest,buckets),file_name=f'{kind}_review_responses.csv',mime='application/octet-stream')
    key=f'cohort_item_{kind}_{order}'
    prioritized=kind=='glyph' and order=='uutien'
    pending=glyph_order(rows,latest,quality,counts,hints) if prioritized else [i for i in range(1,len(rows)+1) if i not in latest]
    advance=st.session_state.pop('cohort_next',None)
    if advance and advance[0]==key: st.session_state[key]=advance[1]
    if key not in st.session_state: st.session_state[key]=pending[0] if pending else 1
    def jump(target): st.session_state[key]=target
    nav=st.columns(3); current=st.session_state[key]
    nav[0].button('Ảnh trước',disabled=current==1,on_click=jump,args=(max(1,current-1),))
    nav[1].button('Ảnh sau',disabled=current==len(rows),on_click=jump,args=(min(len(rows),current+1),))
    next_pending=(pending[0] if prioritized else next((i for i in pending if i>current),pending[0])) if pending else current
    nav[2].button('Đến mẫu chưa làm',disabled=not pending,on_click=jump,args=(next_pending,))
    idx=int(st.number_input('Mã mẫu',min_value=1,max_value=len(rows),step=1,key=key))
    r=rows[idx-1]; prior=latest.get(idx,{})
    if notice:=st.session_state.pop('cohort_saved',None): st.success(notice)
    st.caption(f"{r['parent']} · {'Đã lưu: '+DECISIONS[kind][prior['decision']] if prior else 'Chưa lưu'}")
    try:
        audit=metadata()[r['parent']]; im=source_image(r,audit)
        if grouped:
            st.subheader('Các glyph trong cùng ảnh')
            reviewer=st.text_input('Bí danh người rà soát',key='cohort_reviewer',max_chars=60)
            ids=[i for i,row in enumerate(rows,1) if row['parent']==r['parent']]
            grouped_review(im,ids,rows,latest,key,db,reviewer,pending)
            return
        if kind=='glyph':
            box=r['pixel_xywh']; roi,_=crop(im,box)
            st.subheader(f"Nhãn nguồn: {r['character']}")
            st.image(roi,channels='BGR',width=144,caption='Vùng ký tự gốc')
            overlay=draw(im.copy(),[[r['character'],*box]],(0,220,0))
        else:
            overlay=draw(im.copy(),[['',*b[1:]] for b in audit['boxes']],(0,220,0))
            st.subheader(f"{len(audit['boxes'])} box trong annotation nguồn")
        st.image(overlay,channels='BGR',use_column_width=True,caption='Annotation nguồn · xem từng ký tự, kể cả vùng không có box')
        if st.checkbox('Xem ảnh gốc không khung'): st.image(im,channels='BGR',use_column_width=True)
    except (OSError,ValueError,KeyError) as exc:
        st.error(str(exc)); return
    reviewer=st.text_input('Bí danh người rà soát',key='cohort_reviewer',max_chars=60)
    with st.form(f'{kind}_{idx}_{prior.get("revision",0)}'):
        domain=st.selectbox('Domain quan sát',list(DOMAINS),index=list(DOMAINS).index(prior.get('domain','unknown')),format_func=DOMAINS.get)
        error=st.selectbox('Loại lỗi (không bắt buộc)',['',*ERRORS],index=['',*ERRORS].index(prior.get('error_type','')),format_func=lambda e:ERRORS.get(e,'Không ghi loại lỗi'))
        note=st.text_input('Ghi chú (không bắt buộc)',value=prior.get('reason',''),max_chars=500)
        if kind=='detector':
            st.caption('Nếu bộ box không thiếu nhưng có box thừa/lệch, chọn “Đủ box” và ghi loại lỗi. “Đủ box” không phải xác nhận mọi box đúng.')
        choice=None
        for col,(decision,label) in zip(st.columns(3),DECISIONS[kind].items()):
            if col.form_submit_button(label): choice=decision
    if choice:
        try:
            db.save(kind,idx,prior.get('revision',0),choice,domain,error,note,reviewer)
            st.session_state['cohort_saved']=f'Đã lưu mẫu {idx}: {DECISIONS[kind][choice]}'
            # Apply navigation before the next widget is instantiated.
            rest=[i for i in pending if i!=idx]
            target=next((i for i in rest if rows[i-1]['parent']==r['parent']),rest[0] if rest else idx) if prioritized else next((i for i in rest if i>idx),rest[0] if rest else idx)
            st.session_state['cohort_next']=(key,target)
            st.rerun()
        except (ValueError,sqlite3.Error,OSError) as exc: st.error(str(exc))