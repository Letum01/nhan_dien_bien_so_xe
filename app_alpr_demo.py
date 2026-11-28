"""Single ALPR demo: Dual-YOLO + CNN Mix / SVM Fusion / RF Fusion."""
import os
from pathlib import Path
import subprocess

import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from alpr import demo_service
from alpr.demo_ui import draw_ordered_boxes, show
from demo_alpr_core import History, csv_bytes, decode
from demo_fusion import FusionHeads, agreement, load_cnn
from dual_yolo_client import ROOT
from evaluation_summary import load_summary

TITLE = 'HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE — ALPR'
METHOD = 'character_yolo'
BACKENDS = {
    'CNN Mix': 'cnn',
    'SVM Fusion': 'svm_fusion',
    'Random Forest Fusion': 'rf_fusion',
}
ERRORS = (
    OSError, ValueError, RuntimeError, subprocess.SubprocessError,
    Image.DecompressionBombError, Image.DecompressionBombWarning, cv2.error,
)


@st.cache_resource
def cnn():
    return load_cnn()


@st.cache_resource
def fusion() -> FusionHeads:
    return FusionHeads(cnn())


def infer(image, backend):
    """Regression entrypoint: Dual-YOLO then one classifier or the Fusion trio."""
    return demo_service.infer(image, backend, cnn(), fusion)


def download(rows, name):
    st.download_button('Tải CSV', csv_bytes(rows), file_name=name, mime='application/octet-stream')


def dashboard():
    st.subheader('Validation — 12 biển development, crop YOLO Plate')
    try:
        data = load_summary()
    except (OSError, ValueError, KeyError) as exc:
        st.info(f'Chưa có artifact đánh giá hợp lệ: {exc}')
        return
    rows = [{'Model': name, 'Matched Glyph Accuracy': f"{v['matched_correct']}/{v['matched_total']}",
             'Exact Plate Match': f"{v['exact']}/{data['images']}",
             'CER': v['edits']/data['reference_characters'],
             'Model SHA-256': v['checkpoint_sha256']} for name,v in data['models'].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"{data['images']} ảnh / {data['reference_characters']} ký tự tham chiếu. "
               'Development regression đã dùng trong phát triển, không phải test độc lập. '
               'Không phải độ chính xác của ảnh người dùng vừa upload.')


def main():
    st.set_page_config(page_title=TITLE, layout='wide')
    st.title(TITLE)
    st.caption('YOLO Plate + YOLO Character + CNN/SVM/RF Fusion • CUDA detector • một biển score cao nhất mỗi ảnh')
    mode = st.sidebar.radio('Chức năng', ['Ảnh đơn', 'So sánh thuật toán', 'Hàng loạt', 'Lịch sử', 'Đánh giá'])
    if mode == 'Đánh giá':
        dashboard()
        return
    if mode == 'Lịch sử':
        history = History(os.environ.get('ALPR_DEMO_HISTORY', str(ROOT / 'data/alpr_demo_history.sqlite3')))
        rows = history.search(st.text_input('Tìm tên ảnh hoặc biển số'))
        st.dataframe(rows)
        if rows:
            download(rows, 'alpr-history.csv')
        confirm = st.checkbox('Xác nhận xóa toàn bộ lịch sử demo')
        if st.button('Xóa lịch sử', disabled=not confirm):
            history.clear()
            st.rerun()
        return
    backend = BACKENDS[st.sidebar.selectbox('Nhận dạng', list(BACKENDS))]
    save = st.sidebar.checkbox('Lưu lịch sử (không lưu ảnh)')
    batch = mode == 'Hàng loạt'
    compare = mode == 'So sánh thuật toán'
    uploads = st.file_uploader('Chọn ảnh JPG/PNG', type=['jpg', 'jpeg', 'png'], accept_multiple_files=batch)
    files = uploads if batch else ([uploads] if uploads else [])
    st.caption('Tối đa 20 MiB / 24 MP mỗi ảnh; tối đa 20 ảnh mỗi lượt.')
    if not st.button('Chạy nhận diện', disabled=not files):
        return
    if len(files) > 20:
        st.error('Chọn không quá 20 ảnh mỗi lượt.')
        return
    requested = 'compare' if compare else backend
    rows = []
    progress = st.progress(0) if batch else None
    for index, file in enumerate(files):
        try:
            image = decode(file.getvalue())
            if not batch:
                show(image, 'Ảnh gốc')
            with st.spinner('Đang xử lý…'):
                result = infer(image, requested)
            keys = result['text'] if compare else [backend]
            rows.extend(demo_service.result_row(file.name, METHOD, key, result) for key in keys)
            if not batch:
                render(image, result)
        except ERRORS as exc:
            rows.append(demo_service.error_row(file.name, METHOD, backend, exc))
            st.error(f'{Path(file.name).name}: {exc}. Kiểm tra CUDA/checkpoint/worker log; không chuyển CPU.')
        if progress:
            progress.progress((index + 1) / len(files))
    if save:
        History(os.environ.get('ALPR_DEMO_HISTORY', str(ROOT / 'data/alpr_demo_history.sqlite3'))).save(rows)
    st.dataframe(rows, hide_index=True)
    download(rows, 'alpr-results.csv')


def render(image, result):
    plate = result['plate']
    if not plate:
        st.warning('Không phát hiện biển số ở ngưỡng 0,4.')
        return
    overlay = image.copy()
    x, y, width, height = plate['bbox']
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
    left, right = st.columns(2)
    with left:
        show(overlay, f'Box biển — score detector {plate["score"]:.3f}')
    x, y, width, height = plate['crop_box']
    crop = image[y:y + height, x:x + width].copy()
    draw_ordered_boxes(crop, result['glyph_boxes'])
    votes, trio = agreement(result['text'])
    glyphs = []
    for index, (gx, gy, gw, gh) in enumerate(result['glyph_boxes']):
        symbols = {key: text[index] for key, text in result['text'].items() if index < len(text)}
        glyphs.append({
            'Thứ tự': index + 1,
            'Box': str([gx, gy, gw, gh]),
            **symbols,
            'Đồng thuận': f'{votes[index]}/3' if index < len(votes) else None,
            'Score box': result['glyph_scores'][index] if result['glyph_scores'] else None,
        })
    with right:
        show(crop, 'Crop và thứ tự ký tự')
    st.subheader('Chuỗi dự đoán')
    st.write(result['text'])
    st.dataframe(glyphs)
    if trio:
        st.caption(f'Đồng thuận là số phiếu, không phải xác suất đúng. So sánh: {" / ".join(trio)}.')
    disagree = [index for index, vote in enumerate(votes, 1) if vote < 3]
    if disagree:
        joined = ', '.join(map(str, disagree))
        st.warning(
            f'Không chắc chắn: bất đồng {len(disagree)}/{len(votes)} vị trí ({joined}). Không tự sửa chuỗi.'
        )
    if not glyphs:
        st.warning('Có vùng biển nhưng không tách được ký tự.')
    st.json(result['timings'], expanded=False)
    st.caption(
        f"Worker PID {result['worker']['pid']} · {result['worker']['gpu']} · "
        f"{'cold' if result['cold'] else 'warm'} · {result['timings']['total']:.3f} s"
    )


if __name__ == '__main__':
    main()
