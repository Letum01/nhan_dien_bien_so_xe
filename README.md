# Nhận diện biển số xe Việt Nam (ALPR)

## Chạy bản demo nghiệm thu trên Windows

Mở **`START_ALPR.bat`** tại thư mục dự án và truy cập **http://localhost:8511**. BAT kiểm tra `.venv`, `.venv-yolo`, hash checkpoint/protocol, CUDA và cổng. Giữ cửa sổ chạy trong lúc trình diễn. Demo dùng Dual-YOLO + CNN Mix ban đầu; ứng dụng production cũ không đổi.

- [Kiến trúc](docs/ARCHITECTURE.md) · [Hướng dẫn](docs/USER_GUIDE.md) · [Huấn luyện và dữ liệu](docs/TRAINING_AND_DATASET.md)
- [Đánh giá](docs/EVALUATION_REPORT.md) · [Giới hạn](docs/LIMITATIONS.md) · [Báo cáo bàn giao](reports/ALPR_FINAL_PROJECT_REPORT.md)

Phần dưới là nhật ký các mốc phát triển trước đây; dùng báo cáo bàn giao cho trạng thái mới nhất.

## Hard15 frozen benchmark đã chạy

Mốc40OCR/327glyphs xácnhận(train270,val57,test0). Đủ15protocolhardIDs/123glyphs,
5biểnmới40glyph. Khôngđổiprotocol/config. GTcropbaselineF1.7164/exact5/15→
borderF1.9060/exact7/15;YOLOpad15baselineF1.7742/exact3/15→borderF1.8675/
exact8/15. CropGT0409border mất4glyph nênkhôngpromoteproduction.
Reports/overlays/contactsheet:
reports/glyph_audit_20261125T103134292702Z/HARD15_FINAL_REPORT.md vàhard15_final.json.
130pytestPASS,validator327PASS,browserhardqueuePASS. Train-development,
khôngindependenttest;khôngchọnlạiparameters,tránhclaimaccuracytổngquát.

## Chốt benchmark mốc15 hardcases

`reports/hard15_protocol.json` đã khóa first15queueIDs,weights/codehash,
cropGTkhôngpadding/cropYOLOpad15%,baselinevborder,IoU.5. Khôngchọntheoval.
`hard15_benchmark.py run` hiệnBLOCKEDvì10/15reviewed;khôngchạyinferencepartial.
Sau5ngườiduyệtthật,mởrunđểaudit/exportrồi benchmark fixed. Không overwriteprotocol
để néhashconflict. Xemreports/OCR_HARD_MILESTONE_40_PENDING.md/json.

## OCR mốc35 và bộ đếm snapshot

SQLite35plates287glyphs=train230+val57,test0;hard10/28done. Counter mismatch
222trainwasstale/mixedrenderpossible:header/progress/hardusedseparateall_latest
readsandsaveatbottomwithoutrefresh. Nowone reviewed snapshot supplied toboth
progress/hard andimmediate st.rerun afterreviewedsave. No labels changed.
Original mismatchingframeeventtimeline cannotbe recovered. Synthetic temporary
Chromiumsave/reloadcountertestPASS. Report:reports/OCR_HARD_MILESTONE_35.md.
Artifacts:reports/glyph_audit_20261125T095817713548Z/,newhard83raw/normalized
contactsheet,classsplitCSV,validator287PASS. No retraining or segmentationpromotion.

## OCR Hard Cases — ưu tiên

```powershell
$env:ALPR_OCR_QUEUE='hard'
.\.venv\Scripts\python.exe -m streamlit run label_characters.py --server.port 8505
```

Hoặc chọn hàng đợi **OCR Hard Cases — ưu tiên**. Dùng queue28train trong
reports/ocr_hard_queue_20261125T091023673153Z/queue.json,kiểmtra nguồn/hash/
Detectionreviewedrevision/box/group/split. ĐãGlyphreviewedthìẩnkhỏichờ,khôngxóa.
Mụclỗiparentbịbáoriêng,khôngtựđổinhãn. Dashboard:remaining/completed/train
glyphs,classmissing/1–2mẫu;JSONlýdoưu tiênnhỏ/mờ/nghiêng/biênlàheuristic.
Ảnhkhôngcóproposalphảivẽthêm,máykhôngtựduyệthoặcbỏảnhkhó.

Mốc10và15hardplatesreviewedtrongqueuechạyaudit/export/validator/contactsheet/
classcountsplitmộtlầnmỗi mốc,khôngtrain/khôngpromotesegmentation. Chỉaudit
revisionđãreviewed,draftautosavekhôngduyệt. Cảnhbáooverlap/order/textmismatch
chặnreviewnhưcũ;mép/nétchỉcảnhbáođểngườikiểmtra.

## Train-only targeted segmentation + hard OCR queue

`segmentation_train_ablation.py` thử trên18TRAINcropsfrozenpad15only;khôngrerun
VALđểchọn. GeometryF1.8927→.9007nhưngFN18→20andexact10/18→8/18;khôngpromote.
ProtectedborderthêmFP;projection_split/row_sortkhôngcải thiện. Fullperplatebox/
classerror/overlapdiagnostics vàoverlaytrong
reports/segmentation_train_ablation_20261125T090913267629Z/REPORT.md.

`prepare_ocr_hard_queue.py` tạo28TRAINDetectionreviewedchưaGlyphreviewed,
khôngđổigroup/split/text/class. reports/ocr_hard_queue_20261125T091023673153Z/
queue.csv/json +crops xếphạngsmall/blur/frameproxiesvàconfirmedtextcoverage.
Điểmmáykhôngphảinhãnreviewed;khôngtựđọcbiểnđểgánclass. Cầnngườiduyệttiếp.

## Crop padding experiment (không promote)

Script `benchmark_crop_padding.py` thử padding0/5/10/15%mỗiphía theoYOLOw/h,
clipbiên,giữbox/scoregốc,baseline vàborder_components. TrainF1chọn15%+border,
khóatrướcVAL. Báocáo reports/crop_padding_20261125T084749402271Z/REPORT.md:
valcoverage54/57→57/57glyph,fullcrop6/7→7/7,F1.8644→.9138,nhưngexacttext
1/7→1/7.4ảnhsegF1tăng,2giảm,1khôngđổi. Khôngthayproductionpadding/segmentation,
weights,threshold.4hoặcdefaultOpenCV. BenchmarkkhôngdùngGTchọncrop/chữ.
124pytestPASS,ChromiumStreamlitupload/switchOCR/DiagnosticsPASS.

## Segmentation benchmark (thử nghiệm, không đổi production)

`segmentation_benchmark.py` dùng reviewed glyph boxes,one-to-oneIoU>=.5
maximum-cardinality,khôngghépindex. Chọnvariantbằng18trainplatesF1 rồiđánhgiá
7valplates. `border_components` chọntrêntrain:trainF1.8271→.8968,val.8908→.9138;
fullmatch/order3/7→5/7,GTcroptext3/7→4/7. Mộtvariantkháctốtvalhơnkhôngđược
chọnlạibằngval. Giữmọivariantngoàiproduction. Annotator đãdùngproposalOpenCV,
nhiềumatchedIoU1.0cóthểphảnánhannotationbias,khôngphảiindependenttest.

ExperimentalYOLO→selectedsegment→CNNmixexact2/7developmentval;khôngGTrepair.
Classifier50/53matchedglyphphảibáokèmsegmentation53/57,khôngtínhmissingasđúng.
Artefactsreports/segmentation_benchmark_20261125T083659193188Z/:REPORT.md,
benchmark.json/comparison.csv/rates.json,overlay/binary/contourreason,
matchedclassifierconfusion,errorbreakdown,endtoendoverlays. Khôngtrainclassifier,
khôngđổithreshold/defaultdetector/labels/weights.122pytestPASS,browsersmokePASS.

## CNN paired experiment02 đã chạy (chưa promote)

Frozenbundle3600synthetic/147realtrain/57realval,CPU TensorFlow2.16.2. Cùngseed42,
initialweightsSHA/architecture/Adam.001/batch16/max30/patience5,trainonlyaugmentation.
Synthetic-onlybestepoch5/actual10 đạt29/57;mixbestepoch14/actual19 đạt54/57.
CandidateMajoritySVMmix/RFmix/CNNmix52/57,khôngtốthơnCNNmixnênkhôngthayproduct.
Validation7plateskhôngphảifinaltest; no independentOCRtest. Auto segmentation
trênGTcrop vẫn dư/nhiễu:14.jpg8reviewed→9auto;0408reviewed9→11auto.
Không coi54/57glyphlàplateaccuracy hoặcALPRendtoendaccuracy.

Weightsnew:experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/;
learningcurves,confusion,perglyph/perplate/group và errorlogs tại
reports/OCR_MIX_EXPERIMENT_02.md/json. Baseline/YOLO/splits/threshold/defaultdetector
khôngđổi. Resumeexperimentchỉreusesrunhoàntấtsauconfiguration/hashcheck,không
trainđè. Không tựchạy thêmepochđểtốihóa57valglyphs.

## OCR mix experiment01 đã chạy (SVM/RF, không promote sản phẩm)

Đúngbundle3600synthetic+147realtrain,57realval;preflightregeneration/hash/live
reviewedparent/leakagePASS. Kết quảdevelopment57glyphs/7plates:SVMonly3/57,
SVMmix49/57;RFonly7/57,RFmix50/57. Macrorecallsupported17classes tăng vàIcollapse
giảm mạnh;không gọi là independenttestaccuracy. Realtrain thiếu15classes,
noOCRtest. Oldbaseline provenance không đầy đủ. Weightsproduk không thay.

Artifactnewruns:experiments/ocr_mix_experiment01_20261125T072726619780Z/.
Fullmetrics/perplate/class/confusion/fitseconds:reports/OCR_MIX_EXPERIMENT_01.md
vàJSON. Báocáobiểuđồrenderbằng.venv-yolo vì mainvenvkhôngcómatplotlib:

```powershell
.\.venv-yolo\Scripts\python.exe report_ocr_mix_experiment01.py
```

CNNchưatrain trongthínghiệmnày. ThửCNNcần giữfrozenvalidation/config/newrun,
không điều chỉnh test hoặc tự thay model sản phẩm. Chưa calibration score.

## Chuẩn bị synthetic + real OCR (chưa train corpus thật)

Bundle cố định `experiments/ocr_mix_bundle_20261125/`:3600synthetictrain,
147realtrain,57validation giữ riêng. Sourcehash/seed42/mapping/preprocessing
manifest lưu cùng arrays.npz. Nhãn synthetic từ lệnh vẽ, không từprediction.
Provenance chính xác của weights cũ không có manifest nên UNVERIFIED.
`ocr_mix_experiment.py prepare` không training; `fit` là thao tác tường minh,
newrunfolder không ghi đè baseline. Config synthetic-only/mix50%domainweight/
CNNfine-tune tách riêng. `ocr_mix_ensemble.py` chỉ combine run có cùng validation
manifest/order. Không softvote từ score chưacalibrate.

Xem **reports/OCR_MIX_READINESS.md** cho readiness, rủi ro và lệnh chạy đề xuất.
SVM/RF/CNNfixturetraining được kiểm tra bằng pytest, nhưng chưa fit corpus thật.
READYcho thử nghiệm khám phá sau quyết định người dùng; không phải deployment
ready hay độc lập finaltest. Không dùngvalđểfit/augment/scaler.

## OCR Train — ưu tiên (15 biển tiếp theo)

```powershell
$env:ALPR_OCR_QUEUE='train'
.\.venv\Scripts\python.exe -m streamlit run label_characters.py --server.port 8504
```

Hoặc chọn **Hàng đợi OCR → OCR Train — ưu tiên**. Queue đóng băng tại
reports/ocr_train_priority.json, lấy đúng15gợi ý của milestone,7textconfirmed
đứng đầu. Không tự điền nhãn glyph. Parent group/split/revision/box phải khớp
queue; thay đổi sẽ báo lỗi để audit thay vì tự cập nhật. Split luôntrain theo
nguồn. Hiển thị tiến độ15biển mới,trainplate/glyph,class thiếu/1–2mẫu và bảng
train/validation riêng. Không chuyển val để làm đẹp độ phủ train.

Contoured frame/noise:chọn index→xóa;rectvẽ thêm;transform sửa;split/merge cùng
dòng;nhập nhãn/line và xác nhận. Chỉ LưuGlyphreviewed sau validator và xác nhận
cuối. Draftautosave không tự duyệt. Khi cả15queue đãreviewed, refresh UI sẽ
audit toàn corpus,contactsheet,từngclass/split và validator; signature chặn
lặp với cùngrevision. Hiện queue0/15,không tự tạo nhãn để test mốc hoàn thành.
Không trainingclassifier trong luồng này. Chromium trainqueueproposal/delete/
reloaddraftPASS vớiDBtạm; reviewedpersistence unit-testedfixture,không tạo nhãnthật.

## OCR coverage và validation tiếp theo

Giao diện glyph nay mặc định ưu tiên validation chưareviewed, độc lập group/
hash/pHash với nguồn OCRtrain đãreviewed. Bỏ giới hạnpilot20theo mặc định vì
pilot cũ19train/1val. Không đổi group/split hoặc gán nhãn tự động.
Dashboard hiện bảng glyph/plate từng split,36class counts,missing/low(<5)mẫu.
Mốc>=10reviewedplate tự tạo audit/exportvalidator một lần cho mỗi signature
revision khi apprefresh (không training). Lỗi validator hiển thị trong kết quả.
Contact sheet được tạo bởi audit_glyph_reviews.py trong thư mục mới theo timestamp.
Mép/nét heuristic chỉ cảnh báo: nhiều glyph crop sát nét sẽ chạm biên bình thường,
không tự loại hoặc sửa. Kiểm tra original và normalized sheet bằng người.

```powershell
.\.venv\Scripts\python.exe glyph_progress.py
.\.venv\Scripts\python.exe audit_glyph_reviews.py
```

## Gắn nhãn OCR / Glyph — kho riêng, không sửa detection

```powershell
.\.venv\Scripts\python.exe prepare_glyph_pilot.py
.\.venv\Scripts\python.exe -m streamlit run label_characters.py --server.port 8503
```

Hoặc chọn chế độ **Gắn nhãn OCR / Glyph** trong app gắn nhãn detection. Chỉ lấy
crop từ latest Detection reviewed; ưu tiên11text confirmed và pilot20biển trong
reports/glyph_pilot.json (không tự duyệt glyph). OCR draft/revisions nằm riêng
data/ocr_glyphs.sqlite3; image/source/plate identity, detection revision, group
và split giữ nguyên. Không chỉnh reviewed detection hoặc test/pilot khi gắn OCR.

1. Chọn crop, xác nhận/sửa text bằng người và bố cục1/2dòng, không dùng prediction.
2. Tạo glyph proposal OpenCV CHƯA NHÃN. Viền/ốc vít: chọn index→xóa; glyph thiếu:
   rect vẽ; transform kéo/resize; split offset/merge cùng dòng khi cần.
3. Nhập từng nhãn0-9/A-Z, đặt dòng0trên/1dưới, xem phóng to, xác nhận đủ nét.
4. Sắp xếp dòng/trái→phải; chuỗi/count phải khớp text sau bỏ separator (không
   đổiB/8,O/0). Box ngoài crop/chồng lấn>35%/class sai/thiếu xác nhận bị chặn.
5. Tích xác nhận cuối rồi Lưu Glyph reviewed. Draft autosave riêng, không tự duyệt.
6. Export sang thư mụcMỚI: ảnh gốc/normalized32x32 theo split nguồn, CSV chung
   characters.csv và revision JSON. Chạy validate_glyph_dataset.py trước train.

```powershell
.\.venv\Scripts\python.exe validate_glyph_dataset.py datasets/glyph_round01
.\.venv\Scripts\python.exe browser_check_glyph_review.py
```

ValidatorPASS không đảm bảo đủ class/samples. Hiện0glyph thật reviewed, chưa
train classifier. Test Chromium mới kiểm tra render/proposal/delete/reload và
chặnreview chưa đủ, không tạo reviewed thật. Các mouse split/merge fixture ở
browser_check_glyphs.py là kiểm thử riêng; chưa kiểm thử lại toàn bộ mouse path
trong giao diện OCR riêng này. Không có accuracy glyph chưa đo.

## OCR Diagnostics — trace thực tế, không thay preprocessing sản phẩm

Upload ảnh, chọn YOLO vòng2 hoặc giữ detector mặc định, bấm **OCR Diagnostics —
YOLO vòng2 (lưu artifact)**. Diagnostics luôn dùng YOLO vòng2 cố định.4, tạo
reports/ocr_diagnostics/TIMESTAMP_IMAGEID/ mới với crop/gray/CLAHE/binary,
contour kept/rejected+lý do, reading order, glyph raw/normalized, top3/score
thô của SVM/RF/CNN và Ensemble. Mọi score chưa hiệu chuẩn trên ký tự thật.
Lưu ảnh chỉ khi bấm diagnostics; chú ý dữ liệu cá nhân trong reports khi chia sẻ.

```powershell
.\.venv\Scripts\python.exe ocr_diagnostics.py dataset/license_plate_dataset/0/images/14.jpg
.\.venv\Scripts\python.exe probe_character_models.py
```

14.jpg baseline21contours→11glyph có3mảnh frame, CNNIIIIIIIIIII. Adaptive thử
nghiệm41contours→8glyph nhưng CNNvẫnIIIIIIII. Không bật adaptive tự động:
chưa có glyph ground truth. SQLite có11text confirmed nhưng0glyph confirmed;
không có accuracy ký tự thật. Probe synthetic chỉ kiểm tra plumbing, không
đại diện recognition thực. Xem reports/ocr_diagnostics/FINAL_DIAGNOSTIC_REPORT.md.

## Thử YOLO vòng 2 trong ứng dụng

Chạy app bằng venv chính, chọn **YOLO vòng 2 — thử nghiệm** trong Detector.
OpenCV vẫn mặc định. Tùy chọn dùng checkpoint splitB và Python ở .venv-yolo;
không cần cài Ultralytics vào venv TensorFlow. Threshold cố định0.4, hiển thị
tất cả bbox/score model; OCR xử lý một crop và không khẳng định text chính xác.
Cache24ảnh theo checkpoint/mtime; worker YOLO khởi tạo trên cache miss, không
tuyên bố realtime. Thiếu môi trường/checkpoint báo lỗi rõ ràng.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe browser_check_yolo_product.py
```

Browser test đã upload14.jpg và đổi4OCR, không exception. Báo cáo/screenshot:
reports/round2_streamlit/. Trường hợp AQUA2_20449:box score.708 IoU.466 không
bao đủ extent GT, còn boxIoU.678 score.125 bị lọc ở.4. Không sửa nhãn hoặc hạ
threshold theo validation này. Fixture nhiều xe/ảnh đen/biển thu nhỏ chỉ là test
runtime, không phải accuracy dữ liệu thực.

## Kết quả thực thi phương án B và training vòng 2

User đã xác nhận lại đúng33 thay đổi thực tế:31val→train và2train→val (mô tả
cũ "33val→train" không đúng). SQLite migration transaction tạo33revision mới,
giữ nguyên box/source_group/text/glyph và toàn bộ21pilot. Backup/receipt:
`reports/splitB_applied_20261124T154151235536Z/`. Chạy lặp operation có receipt
không tạo revision trùng; xung đột snapshot rollback/chặn. Draft giống reviewed
cũ được cập nhật base; draft khác giữ nguyên và sẽ yêu cầu xử lý conflict.

Snapshot46/12/3 đã validator/quality/leakagePASS:
`reports/round2_gate_20261124T154153249700Z/dataset`.
Fine-tune từ pretrained gốc, không từ pilot, early-stop epoch57 (best37):
`experiments/yolo_round2_splitB_run01/weights/best.pt`.
Training CUDA RTX2060,batch4,imgsz640,max100,patience20,seed42,workers2,AMP.
Không chạy test (`--skip-test`); không đổi detector/threshold mặc định.

Trên cùng12validation không thuộc train của hai model:conf.4 pilot0TP/0FP/12FN,
vòng2 11TP/1FP/1FN. Đây là validation chọn checkpoint, KHÔNG là test độc lập.
Các33ảnh chuyển sang train không được tính validation. Không tuyên bố chất
lượng tổng quát từ61ảnh. Báo cáo đầy đủ: `reports/SPLIT_B_FINAL.md/json`;
ảnh/threshold/confidence: `experiments/yolo_round2_splitB_run01/validation_comparison/`.
ALPR crop→segmentation→OCR smoke không lỗi nhưng không có OCR accuracy mới.

## Quality gate và phương án split vòng 2

0408 reviewed v3 đã bỏ box2x2; snapshot
`reports/round2_gate_20261124T152339309470Z/dataset` validator PASS17/41/3.
Không tự chạy100epoch khi chỉ17train. `propose_round2_splits.py` tạo báo cáo
A/B chỉ đọc SQLite; không đổi reviewed/split. Phương án hiện lưu trong
`reports/split_options_20261124T152525043845Z/`: A thêm63train→80/41/3 hoặc
khuyến nghị+80train→97/41/3; B đề xuất46/12/3 bằng nhóm nguồn/cụm hash seed42,
giữ toàn bộ pilot. B NOT_APPLIED, cần người dùng xác nhận và revision mới.
Không coi validation chuyển sang train còn độc lập với các phân tích cũ.

UI vòng2 có countTRAIN/VAL, cảnh báoVAL>TRAIN, hiển thị split nhóm đã có và
checkbox xác nhận riêng group/split trước duyệt. Không tự gán. SQLite không
ghi click intent nên không thể kết luận việc chọn nhiềuval là chủ động hay
vô tình. Không có code tự phân bổval trong UI hiện tại.

## Gắn nhãn vòng 2 — 120 ảnh

```powershell
$env:ALPR_ANNOTATION_ROUND='2'
$env:ALPR_ROUND_ONLY='0'
.\.venv\Scripts\python.exe -m streamlit run label_detection.py --server.port 8502
```

Hoặc chọn **Chế độ gắn nhãn → Gắn nhãn vòng 2** ngay trong app. Danh sách lấy
đúng candidates.csv, kiểm tra file/hash/decode; thứ tự theo rank. Dashboard
phân biệt chưa làm/draft/reviewed/cần sửa; nút tải CSV tiến độ và cần sửa không
chỉnh dữ liệu. Autosave chỉ lưu draft. Chỉ cần box, group thực tế, split train/val
phù hợp nhóm nguồn và xác nhận box; không bắt buộc text/glyph. Bật tùy chọn sang
ảnh tiếp theo để tự chuyển SAU KHI lưu reviewed thành công, không tự duyệt ảnh.
Split khởi đầu UNASSIGNED; không sao chép source_group gợi ý thành nhãn người.

Trước khi xác nhận: kiểm tra group hiện có/đóng băng, exact/pHash leakage.
Vòng 2 không cho sửa ảnh pilot hoặc gán ảnh mới vào test. Export kiểm tra giữ
nguyên box/group/split của 21 ảnh pilot; phát hiện thay đổi sẽ báo lỗi chứ không
tự sửa lại. Test pilot vẫn đóng băng và không dùng để chọn threshold. Không có
lệnh training tự chạy từ UI này.

Kiểm tra mới nhất: 120 đường dẫn/hash/decode hợp lệ; 103 chưa làm, 11 draft,
6 cần sửa, 0 reviewed (số liệu tại thời điểm triển khai, dashboard đọc SQLite
mới nhất). Chromium render/điều hướng đã test với DB tạm; approve-auto-next
chưa kiểm thử browser bằng nhãn thật. Xem reports/round2_plan/browser/.

## Duyệt nhanh hàng đợi 150 ảnh

```powershell
.\.venv\Scripts\python.exe prepare_round_proposals.py
$env:ALPR_ROUND_ONLY='1'
.\.venv\Scripts\python.exe -m streamlit run label_detection.py
```

Script giữ nguyên hàng đợi, tạo preview proposal vàng và index trong
`reports/round_01_proposals/`, không tạo reviewed/source_group/split. Hiện có
145 ảnh có gợi ý và 5 ảnh không có; box chỉ là gợi ý, có thể sai/thiếu biển.
Trong giao diện: chọn index proposal → đưa vào canvas → sửa/xóa/thêm box →
nhập source_group thật và chọn split (ban đầu UNASSIGNED) → xác nhận rõ ràng.
Không coi cluster gần trùng là source_group đã xác minh. Không chuyển nhãn từ
ảnh khác sang ảnh đang xem. Tab lịch sử/xuất có nút kiểm tra reviewed/export;
đủ split + không lỗi/leakage mới hiện EXPORT_READY, vẫn cần chạy validator của
snapshot trước training và đánh giá chất lượng/số lượng mẫu.

## Vòng gắn nhãn 1 và môi trường GPU đã kiểm tra

`reports/annotation_round_01.csv/json` chứa 150 ảnh ưu tiên, một đại diện mỗi
cụm exact/pHash gần trùng, luân phiên nguồn/độ phân giải, ưu tiên 0405/0407.
Không phải ground truth hay split huấn luyện. Bật **Chỉ hàng đợi vòng 1** trong
giao diện. `prepare_annotation_round.py` cập nhật queue/readiness nhưng không
duyệt nhãn. `reports/training_readiness.json` hiện BLOCKED vì 0 reviewed.

Đã cài `.venv-yolo` riêng: torch2.5.1+cu121, torchvision0.20.1+cu121,
ultralytics8.3.40. Phép nhân CUDA và nano architecture forward640 chạy trên
RTX2060; báo cáo `reports/yolo_cuda.json`. Không tải hoặc dùng model generic
như detector biển. Không chạy full train khi thiếu nhãn.

```powershell
.\.venv-yolo\Scripts\python.exe verify_yolo_cuda.py
.\.venv\Scripts\python.exe prepare_annotation_round.py
.\.venv\Scripts\python.exe browser_check_glyphs.py
```

Browser glyph dùng ảnh synthetic và database tạm: không phải nhãn người của
dataset chính. Các lỗi/giới hạn resize còn được ghi riêng trong
`reports/glyph_browser/results.json`; xem `reports/ROUND_01_STATUS.md` trước khi
coi công cụ đã nghiệm thu. Component glyph mở bằng nút **Mở canvas ký tự** khi
tab hiển thị; thêm trường index trực tiếp, delete-one. Khởi tạo canvas rỗng
không được coi như xóa draft phục hồi. Full export button trong browser chưa
test; pytest có roundtrip export YOLO/OCR/validator bằng fixture tách biệt.

## Giao diện gắn nhãn tích hợp (cập nhật)

```powershell
.\.venv\Scripts\python.exe -m streamlit run label_detection.py
```

Một giao diện bốn tab: box → text → glyph → lịch sử/xuất. `label_characters.py`
cũng mở giao diện này; `label_crops.py` giữ trình nhập text CSV độc lập cũ.
Dashboard đếm theo ảnh nguồn (kể cả bản sao hash); Reviewed detection không có
nghĩa đủ điều kiện train OCR. Tìm tên/lọc trạng thái, ảnh trước/sau, zoom và
khôi phục ảnh cuối được hỗ trợ. Chỉ dùng nhãn reviewed mới nhất khi export;
draft sửa một nhãn reviewed không âm thầm thay ground truth.

1. Chọn ảnh; rect vẽ, transform kéo/resize; xóa bằng toolbar hoặc nút xóa tất
   cả. Gợi ý OpenCV chỉ có trạng thái proposed, không duyệt tự động.
2. Nhập group và split, kiểm tra đầy đủ box rồi xác nhận detection.
3. Tab text: nhập nguyên chuỗi người đọc, số dòng; ký tự không đọc được không
   đủ điều kiện OCR. Xác nhận text một cách tường minh.
4. Tab glyph: tạo proposal chưa nhãn hoặc tự vẽ; chọn index và split offset /
   merge cùng dòng; sửa class và dòng 0/1, xem ảnh phóng to, xác nhận từng glyph.
   Split/merge/đổi hình học làm mất nhãn xác nhận để bắt buộc kiểm tra lại.
5. Tab cuối: lưu toàn bộ reviewed revision mới. Phục hồi revision tạo bản
   needs_review mới, không tự phê duyệt lại. Xuất vào thư mục MỚI khi đủ cả
   train/val/test và không có group/hash leakage. Chạy validate_yolo_dataset.py
   trên thư mục license_plates được sinh ra trước training.

Autosave ghi working copy riêng SQLite sau các cập nhật được server nhận và
trước khi điều hướng. Không bảo đảm sự kiện chưa gửi trong browser khi đóng
tab/mất mạng. Draft slot/cursor hiện dành cho một người dùng local; xung đột
reviewed-version được chặn nhưng chưa có tài khoản/draft riêng cho nhiều người.
Danh sách gần trùng chỉ hỗ trợ quan sát/gán group, không sao chép nhãn.

Kiểm thử: `browser_check_annotation.py` dùng database tạm, Chromium headless,
chuột vẽ/move/resize, delete-all, autosave, trước/sau/reload; không tạo nhãn thật.
Screenshot và kết quả ở reports/annotation_browser/. Glyph split/merge được
unit test; tương tác glyph canvas/split-merge bằng browser chưa kiểm thử.
Playwright là dependency kiểm thử tùy chọn, không bắt buộc để chạy app:

```powershell
.\.venv\Scripts\python.exe -m pip install playwright==1.48.0
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe browser_check_annotation.py
```

## Dataset audit và gắn nhãn (không tự xác nhận thay con người)

Workspace thực tế: `C:\Users\Admin\Documents\nhan_dien_bien_so_xe`.

```powershell
.\.venv\Scripts\python.exe dataset_audit.py
.\.venv\Scripts\python.exe inspect_training_environment.py
.\.venv\Scripts\python.exe -m streamlit run label_detection.py
.\.venv\Scripts\python.exe -m streamlit run label_crops.py
.\.venv\Scripts\python.exe -m streamlit run label_characters.py
```

Audit hiện tại: 2932 JPG đọc được, 0 ảnh hỏng, 78 nhóm SHA-256 trùng và 864
cặp pHash Hamming <=6 cần kiểm tra. Không xóa ảnh. `reports/dataset_audit.*`
liệt kê từng ảnh và nhóm/cặp; các nhóm xe/ánh sáng/nghiêng/nhiều xe là UNKNOWN
khi thiếu metadata. Chưa có box/text/glyph reviewed đủ điều kiện huấn luyện.

Detection canvas dùng streamlit-drawable-canvas 0.9.3/Streamlit 1.36: rect vẽ
box, transform di chuyển/resize; công cụ canvas undo/delete. Tỷ lệ hiển thị
được quy đổi về pixel gốc (không hỗ trợ xoay rectangle). Proposal OpenCV hiển
thị riêng màu vàng, không tự trở thành nhãn reviewed. Chọn source_group/split,
xác nhận bằng người rồi lưu. **Lưu draft trước khi chuyển ảnh**; các nét vẽ
chưa bấm lưu không được bảo đảm sau khi đóng trình duyệt. Mở lại ảnh sẽ nạp
bản đã lưu từ SQLite. Hiện chưa có auto-next/auto-resume con trỏ ảnh.

`data/annotations.sqlite3` lưu các revision bất biến, thời gian UTC, version,
trạng thái và payload; chống ghi đè phiên sửa cũ. Chỉ reviewed mới xuất sang
reviewed_detection.json. Đổi sang rejected/needs_review loại ảnh khỏi manifest
reviewed nhưng không xóa lịch sử. CSV text cũ được snapshot vào annotation_history
trước khi sửa; text người nhập giữ nguyên, dấu ?/không đọc được để needs_review.

Glyph editor hiển thị từng crop, nhập/sửa bbox JSON, line/order và class label;
thêm/xóa phần tử để tách/ghép, không tự gán class từ chuỗi biển. Khi xác nhận,
xuất nguyên ảnh glyph + normalized 32x32 + CSV vào thư mục MỚI. Glyph bbox hiện
sửa bằng số, chưa có canvas kéo/thả. Kiểm tra glyph không mất nét, polarity
trắng-trên-đen; đánh dấu invert khi cần. Không dùng mảnh noise làm train.

```powershell
# Chỉ sau khi có nhãn reviewed đủ train/val/test:
.\.venv\Scripts\python.exe label_detection.py --export datasets/license_plates
.\.venv\Scripts\python.exe validate_yolo_dataset.py datasets/license_plates
```

Exporter kiểm tra box trùng/out-of-bounds, source_group, exact/pHash leakage
giữa split và từ chối khi phát hiện candidate gần trùng chưa giải quyết.
`data.yaml` và manifest được lưu cùng dataset; không tạo dataset rỗng giả hợp lệ.
`group_split_plan` tạo kế hoạch 70/15/15 seed=42, không sửa nhãn/test đang có;
phải gộp nguồn/cặp gần trùng và validate trước khi chấp nhận kế hoạch.

`configs/yolo_rtx2060.json` chuẩn bị batch8/imgsz640/epochs100/patience20/AMP/
workers2/seed42/device0. Dùng --config với trainer, giảm batch4 nếu OOM có log.
Trainer chạy validator trước khi import YOLO và từ chối fallback CPU khi CUDA
không khả dụng. PyTorch/Ultralytics chưa cài; CUDA runtime torch chưa kiểm tra.
TensorFlow native Windows 2.16.2 hiện GPU=[], không giả định dùng RTX2060;
chưa thay đổi WSL2/hệ thống. Training/resume GPU thực tế vẫn chưa kiểm chứng.

## Cấu hình sản phẩm hiện tại: YOLO/OpenCV + SVM/RF/CNN/Ensemble

**Quy định mới ưu tiên hơn các mục Paddle nghiên cứu bên dưới:** ứng dụng sản
phẩm không dùng PaddleOCR/EasyOCR/Tesseract. Mã Paddle và báo cáo cũ được giữ
để bảo toàn baseline nghiên cứu, không xuất hiện trong lựa chọn Streamlit.
`end_to_end_pipeline` mặc định từ chối OCR ngoài; chỉ công cụ nghiên cứu có thể
bật `allow_research_ocr=True` một cách tường minh.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe compare_product.py
.\.venv\Scripts\python.exe -m pytest -q
```

Giao diện chọn OpenCV và SVM/Random Forest/CNN/Ensemble. Chỉ hiện YOLO khi có
dependency ultralytics và biến `ALPR_YOLO_WEIGHTS` trỏ tới checkpoint local đã
được xác minh huấn luyện class biển số. Không tải generic YOLO để suy luận biển.
Nút so sánh chạy bốn phương pháp trên **cùng crop và cùng danh sách glyph**.
Raw OCR được giữ; Majority có bất kỳ vị trí ba phiếu khác nhau thì trả UNKNOWN.
Phiếu đồng thuận không phải confidence xác suất và không chứng minh chính xác.

PyArrow được ghim 17.0.0 để bảng Streamlit tương thích NumPy 1.26.4; bản
PyArrow 26 đã gây lỗi runtime khi render dataframe dù pip check không báo lỗi.

### Ensemble và đánh giá độc lập

`ensemble_recognition.py` hỗ trợ:
- Majority: >=2/3 phiếu, ba nhãn khác nhau → `?` và trạng thái uncertain.
- Weighted: yêu cầu trọng số dương, hữu hạn và nguồn validation; không có trọng
  số mặc định ưu tiên CNN. Ties → uncertain.
- Soft: yêu cầu metadata calibration/validation và matrix xác suất chuẩn hóa,
  ánh xạ đủ 36 lớp; không lấy margin/heuristic làm xác suất. Chưa bật trong UI.

Không phương pháp nào được quảng cáo cải thiện khi chưa đo độc lập. Majority
chỉ là baseline thử nghiệm có cảnh báo và không thay model đơn mặc định.

```powershell
# Chỉ chạy khi có CSV glyph thật đã human_reviewed, cùng source_group không bị chia tập:
.\.venv\Scripts\python.exe evaluate_ensemble.py experiments/real_glyphs/characters_to_review.csv experiments/real_run_01 experiments/ensemble_report.json
```

Script khóa weighted weights từ character accuracy trên validation (rule đã
định trước), sau đó đánh giá test; lưu dự đoán/UNKNOWN và lựa chọn validation.
Soft voting ghi BLOCKED khi chưa có artifact calibration. Đây là metric ký tự,
không tự suy ra Plate Exact Match hay detection accuracy.

### Tình trạng dữ liệu/GPU kiểm tra trực tiếp

Máy có NVIDIA RTX 2060 6GB VRAM; nvidia-smi driver CUDA UMD 13.4. Venv chính
chưa có PyTorch/ultralytics, nên CUDA runtime torch chưa được xác minh. Dataset
không có reviewed_detection.json, reviewed_crops.csv hoặc glyph được duyệt.
**YOLO training, OCR real-data retraining, validation ensemble và test accuracy
đều BLOCKED vì thiếu nhãn**; không chạy synthetic retraining để giả cải thiện.

Giữ công cụ label_detection.py/label_crops.py và exporter group-isolated. Khi
đủ nhãn và PyTorch CUDA phù hợp, fine-tune nano/small từ checkpoint pretrained
local đáng tin cậy, dùng --pretrained/--device/--batch:

```powershell
python train_plate_detector.py experiments/plate_yolo_dataset/data.yaml local_nano.yaml experiments/plate_run_01 --pretrained trusted_local_nano.pt --device 0 --batch 4 --epochs 50
```

Trainer không tải weights ngầm, không ghi đè output, chạy test validation sau
train. Kiểm tra best.pt/last.pt/results.csv và PR/mAP50/mAP50-95 thực tế trước
khi dùng. Chưa tạo best.pt hoặc model OCR mới trong lần nâng cấp này.

Sáu ảnh phát triển: Majority từ chối 0402/0403/0404/0405/0407 do bất đồng;
0406 đồng thuận nhưng chuỗi vẫn sai (`IIF1IIJ1I`). Xem
`debug_outputs/product_ensemble/predictions.json`. Không tính accuracy do chưa
có nhãn được người xác nhận; lỗi 0407 detector đèn vẫn còn.

## Cập nhật detector và Paddle persistent

Paddle nay giữ một worker CPU cho mỗi interpreter/config trong tiến trình app,
load model một lần, khóa tuần tự yêu cầu và cache 32 kết quả theo bytes crop +
script + config. Worker timeout/exit được dừng/reap; lỗi hiện tại được báo rõ,
yêu cầu tiếp theo khởi tạo lại. Không âm thầm retry kết quả lỗi. Khởi động lại
app sẽ mất worker/cache. `OCRResult.timings` tách cold start, inference, decode,
encode, request và cache hit. Raw text không bị thay đổi.

```powershell
.\.venv\Scripts\python.exe compare_ocr.py
.\.venv\Scripts\python.exe benchmark_paddle.py
```

Đo trên máy này: cold start ~7.31 giây, warm inference ~0.43–0.48 giây/crop
biển đọc được; cache hit được đo riêng. Xem `persistent_benchmark.json` trong
`debug_outputs/ocr_comparison/`. Đây không phải FPS video hoặc tuyên bố realtime.
Chi phí detector chưa nằm trong benchmark crop OCR này.

OpenCV được giữ lại; bổ sung proposal từ khung tứ giác kín giúp 0405 cắt đủ
biển và Paddle đọc `65-H6 / 9154`. 0407 vẫn crop nhầm đèn; hiện Paddle trả rỗng.
Không suy ra detection accuracy từ sáu ảnh phát triển này.

### Detector nhiều biển và YOLO tùy chọn

`detection_backends.detect_plates` trả nhiều crop/xywh/score; `recognize_multiple`
trả kết quả OCR từng proposal, giữ nhãn `unverified_proposal` cho OpenCV (không
coi mọi proposal là biển đúng). UI ảnh hiện tại vẫn dùng một kết quả tốt nhất.
`end_to_end_pipeline(..., detector_type="yolo", detector_weights="path.pt")`
dùng adapter plate-only. Không có trọng số phù hợp trong workspace: chưa bật
YOLO mặc định và chưa cài ultralytics. Adapter không tải model generic; yêu cầu
file checkpoint local, class plate-only. Tên lớp không tự chứng minh chất lượng
checkpoint: người cung cấp phải xác minh nguồn và đánh giá trên test độc lập.

```powershell
.\.venv\Scripts\python.exe -m streamlit run label_detection.py
# Gắn TẤT CẢ biển trong ảnh bằng [x,y,w,h], kiểm tra overlay, xác nhận thủ công.
.\.venv\Scripts\python.exe label_detection.py --export experiments/plate_yolo_dataset
# Chỉ khi đủ nhãn train/val/test độc lập theo xe/video và cài ultralytics riêng:
python train_plate_detector.py experiments/plate_yolo_dataset/data.yaml local_architecture.yaml experiments/plate_detector_run --epochs 50
.\.venv\Scripts\python.exe evaluate_detection.py dataset/reviewed_detection.json --output opencv_detection.json
```

Nhập [] chỉ cho ảnh thực sự không có biển; không lấy prediction làm box thật.
Exporter kiểm tra source_group/split, không ghi đè output. Trainer bắt đầu từ
architecture YAML local (`pretrained=False`), không download checkpoint generic,
và đánh giá checkpoint best trên split test. Chưa chạy huấn luyện vì thiếu nhãn.
Evaluator tính precision/recall tại IoU=.5 cho proposal (kể cả false positives),
mean matched IoU, không giả lập mAP. OCR vẫn dùng label_crops.py và
evaluate_labeled.py --crop-mode; chưa có nhãn được người dùng duyệt để tính accuracy.

## PaddleOCR tùy chọn — đã kiểm tra với API 3.3.2

Giữ nguyên venv chính và detector OpenCV. Paddle dùng môi trường riêng vì
PaddleX yêu cầu OpenCV-contrib khác phiên bản với ứng dụng. Không cài
`requirements-paddle.txt` vào venv TensorFlow đang dùng:

```powershell
py -3.11 -m venv .venv-paddle
.\.venv-paddle\Scripts\python.exe -m pip install -r requirements-paddle.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Chọn **PaddleOCR (chuỗi trực tiếp)** ở thanh bên. Backend tải lười các model
chính thức `PP-OCRv5_mobile_det` (phát hiện chữ trong crop, không phải detector
biển số) và `en_PP-OCRv4_mobile_rec`, lưu cache mặc định của PaddleX.
Lần đầu cần Internet. Có thể chỉ định Python worker bằng `ALPR_PADDLE_PYTHON`.
Lỗi dependency/model/timeout được ghi rõ; CNN/SVM/RF không phụ thuộc Paddle.

`ocr_backends.recognize_crop` trả cùng `OCRResult` cho bốn backend: raw text,
text chuẩn hóa, trạng thái, score nếu có, thời gian, tokens và thông tin lỗi.
Raw Paddle giữ nguyên dấu phân cách/xuống dòng; text chỉ bỏ separator, không
thay B/8, O/0. Score Paddle là min điểm các text token, **chưa hiệu chuẩn**.
Worker CPU khởi tạo lại model cho mỗi crop mới (~8–12 giây trên máy kiểm thử),
có cache theo nội dung cho crop lặp lại; không dành cho realtime.

```powershell
.\.venv\Scripts\python.exe compare_ocr.py
.\.venv\Scripts\python.exe verify_paddle.py
.\.venv\Scripts\python.exe -m streamlit run label_crops.py
```

Bảng so sánh, histogram lớp dự đoán, hash crop và thời gian nằm ở
`debug_outputs/ocr_comparison/results.json` và `results.csv`. Crop 0405 thiếu
biển và 0407 là đèn xe: loại chúng khỏi đánh giá chất lượng OCR, không dùng
kết quả trống để kết luận Paddle kém. Những ảnh còn lại cũng phải được duyệt
nhãn trước khi báo cáo accuracy. AppTest upload là kiểm thử phía server với
payload UploadedFile, không phải kiểm thử browser file picker.

### Duyệt nhãn và huấn luyện dữ liệu thật

`label_crops.py` hiển thị crop không kèm dự đoán gợi ý, yêu cầu nhập text,
source_group và split; lưu `dataset/reviewed_crops.csv` sau xác nhận bằng người.
Ảnh cùng xe/video phải cùng group/split. Chỉ lưu crop có đầy đủ biển số.

```powershell
.\.venv\Scripts\python.exe evaluate_labeled.py dataset/reviewed_crops.csv --crop-mode --model paddle --output paddle_evaluation.json
.\.venv\Scripts\python.exe collect_real_characters.py dataset/reviewed_crops.csv experiments/real_glyphs
```

Exporter chỉ sinh **ứng viên ký tự chưa nhãn** và `characters_to_review.csv`.
Không gán nhãn theo vị trí chuỗi biển vì segmentation có thể dư/thiếu. Xem
từng glyph, nhập `class_label` và đánh dấu `human_reviewed`; loại mảnh viền.
Sau khi đủ cả 36 lớp trong train và có validation/test độc lập theo nguồn:

```powershell
.\.venv\Scripts\python.exe train_real_characters.py experiments/real_glyphs/characters_to_review.csv experiments/real_run_01 --epochs 20
```

Trainer từ chối thư mục đã tồn tại/thư mục trong saved_models; kiểm tra group
leakage và lớp, dùng augmentation chỉ ở train, checkpoint/early stopping theo
validation. Chưa chạy huấn luyện dữ liệu thật vì chưa đủ nhãn. Cần so sánh model
mới với model gốc trên cùng tập reviewed test trước khi đưa vào sử dụng.
> **Automatic License Plate Recognition System**  
> Dự án ứng dụng **Thị giác Máy tính (Computer Vision)** và **Học máy / Học sâu (Machine Learning & Deep Learning)** để nhận diện biển số xe tự động theo quy trình đường ống khép kín (End-to-End Pipeline).

---

> **Lưu ý về kết quả:** các chỉ số 100%/96.94% nêu trong tài liệu phiên bản cũ không có artifact/báo cáo test độc lập đi kèm và chưa được xác minh; không xem đó là độ chính xác thực tế. Hãy xóa các con số đó khỏi báo cáo bảo vệ cho đến khi đo lại.

## Trạng thái và phạm vi đã xác minh

Pipeline hiện tại xử lý **một ảnh và một vùng biển số ứng viên** bằng OpenCV contour, sau đó phân đoạn contour và phân loại ký tự bằng CNN/SVM/Random Forest. Chưa có triển khai video, webcam, YOLO, OCR Paddle/EasyOCR hay đánh giá whole-plate có ground truth. Workspace có các file model, nhưng khả năng tải/suy luận cần xác minh trong môi trường Python đã cài dependencies.

##  1. TỔNG QUAN HỆ THỐNG

Hệ thống cho phép người dùng tải lên ảnh phương tiện giao thông (ô tô, xe máy) và tự động trích xuất chuỗi ký tự biển số xe với độ chính xác cao. 

###  Tính năng nổi bật
- **Định vị biển số tự động (Plate Localization)**: Ứng dụng các kỹ thuật xử lý ảnh nâng cao (Black-hat, Sobel-X, Otsu, Canny, phân tích Contour).
- **Phân tách ký tự thông minh (Character Segmentation)**: Tự động tách từng ký tự, lọc nhiễu ốc vít/viền, hỗ trợ cả **biển số 1 dòng và biển số 2 dòng**, chuẩn hóa ký tự về kích thước chuẩn **32x32 pixels**.
- **Đa dạng mô hình nhận diện (OCR Classification)**:
  - **SVM (Support Vector Machine)**: RBF Kernel kết hợp đặc trưng HOG.
  - **Random Forest**: 100 cây quyết định kết hợp đặc trưng HOG.
  - **CNN (Convolutional Neural Network)**: Mạng học sâu đa tầng Conv2D + BatchNorm + Dropout.
- **Hậu xử lý cú pháp biển số Việt Nam (`Domain Heuristics`)**: Tự động sửa các lỗi nhầm lẫn quang học kinh điển (`I` ↔ `1`, `O` ↔ `0`, `S` ↔ `5`, `B` ↔ `8`) dựa trên quy tắc biển số xe Việt Nam.
- **Giao diện Web trực quan (Streamlit GUI)**: Kéo thả ảnh dễ dàng, hiển thị minh bạch các bước xử lý trung gian.

---

##  2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
may_hoc/
│
├── localization.py        # Module 1: Tiền xử lý & định vị khung biển số xe
├── segmentation.py        # Module 2: Nhị phân hóa Otsu & phân tách từng ký tự (32x32)
├── dataset.py             # Module 3: Trích xuất HOG, Data Augmentation & Cú pháp biển số VN
├── models.py              # Module 4: Xây dựng kiến trúc & huấn luyện SVM, RF, CNN
├── main.py                # Module 5: Đường ống End-to-End & đánh giá định lượng Train/Test
├── app.py                 # Module 6: Giao diện web ứng dụng Streamlit GUI
│
├── saved_models/          # Thư mục lưu trữ các mô hình đã huấn luyện
│   ├── svm_model.joblib   # Trọng số mô hình SVM
│   ├── rf_model.joblib    # Trọng số mô hình Random Forest
│   ├── cnn_model.keras    # Trọng số mô hình CNN
│   └── hog_scaler.joblib  # Bộ chuẩn hóa StandardScaler fit trên Train set
│
├── dataset/               # Dữ liệu hình ảnh biển số xe
└── README.md              # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🔄 3. QUY TRÌNH HOẠT ĐỘNG (END-TO-END WORKFLOW)

```text
[ Ảnh xe gốc ]
      │
      ▼
┌──────────────────────────────┐
│       localization.py        │  ==> Cắt khung biển số xe (Plate ROI)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│       segmentation.py        │  ==> Tách các ký tự chuẩn 32x32
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│          dataset.py          │  ==> Trích xuất vector đặc trưng HOG (9 bins, 8x8 cell)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   models.py (RF / SVM / CNN) │  ==> Dự đoán nhãn từng ký tự (0-9, A-Z)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Hậu xử lý cú pháp VN (Rule) │  ==> Sửa nhầm lẫn quang học theo vị trí chuẩn
└──────────┬───────────────────┘
           │
           ▼
    KẾT QUẢ: "51A00530" (Hiển thị lên Streamlit Web)
```

---

## 📊 4. ĐÁNH GIÁ

Chưa có kết quả được xác minh trên dữ liệu biển số thực có nhãn độc lập. Dữ liệu ký tự tổng hợp có thể dùng để kiểm tra pipeline huấn luyện, nhưng không đại diện độ chính xác trên ảnh đường phố. Tập đánh giá đúng cần ground truth box và text, rồi đo precision/recall/IoU, character accuracy/CER, exact plate match và latency.

---

## 🚀 5. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### **Yêu cầu môi trường**
- Python 3.10 hoặc 3.11
- Hệ điều hành: Windows, macOS, hoặc Linux

### **Bước 1: Cài đặt các thư viện cần thiết**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### **Bước 2: Huấn luyện mô hình ký tự tổng hợp (tùy chọn)**
Chạy lệnh sau để tự động sinh dữ liệu, huấn luyện 3 mô hình, lưu checkpoint vào thư mục `saved_models/` và in báo cáo kết quả chi tiết:
```powershell
python main.py
```

### **Bước 3: Khởi chạy Giao diện Web (Streamlit GUI)**
Khởi động ứng dụng web trực quan:
```powershell
python -m streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🛠️ 6. CÔNG NGHỆ SỬ DỤNG
- **Computer Vision**: OpenCV (`cv2`), Scikit-Image (`skimage`).
- **Machine Learning**: Scikit-Learn (SVM, RandomForestClassifier, StandardScaler, Metrics).
- **Deep Learning**: TensorFlow / Keras (Sequential, Conv2D, BatchNormalization, Dropout).
- **Web App GUI**: Streamlit.
- **Model Checkpoint**: Joblib, HDF5/Keras.

### Giới hạn và phát triển tiếp

- Dùng TensorFlow CPU 2.16.2/Keras 3.15 cho tương thích với artifact CNN `.keras`; SVM/RF lưu từ scikit-learn 1.9.0. Không nên nâng riêng các gói này khi đang dùng artifact cũ.
- Không có trọng số detector YOLO đã xác minh. Chỉ tích hợp mô hình được huấn luyện trên bounding box biển số có nhãn, đánh giá trên cùng tập test với OpenCV; không dùng YOLO generic.
- Để tạo tập đánh giá, gắn nhãn box + text bằng người gán nhãn, kiểm tra chéo, chia tách theo nguồn/xe; tuyệt đối không lấy dự đoán mô hình làm nhãn thật.
- Video, webcam trình duyệt, multi-plate end-to-end, OCR hiện đại và UI chuyên biệt cho so sánh chưa được hoàn thành. Localization có proposal nhiều vùng nhưng pipeline vẫn chỉ phân loại vùng tốt nhất.
- Smoke test trên một số ảnh không nhãn cho thấy dự đoán ký tự còn sai rõ rệt; detector contour và classifier huấn luyện ký tự tổng hợp chưa đạt độ tin cậy thực tế. Không trình bày kết quả các ảnh đó như đánh giá accuracy.
- Lịch sử SQLite được lưu tại `data/recognitions.sqlite3`; ứng dụng chỉ ghi khi bấm lưu và không giữ ảnh tải lên.

### Chẩn đoán pipeline theo từng ảnh

```powershell
python debug_pipeline.py --images dataset/license_plate_dataset/0/images/0402.jpg dataset/license_plate_dataset/0/images/0403.jpg --output debug_outputs --model cnn
```

Mỗi ảnh sinh ảnh gốc, edge map, proposal box, crop từng proposal, crop baseline, crop hiệu chỉnh phối cảnh (nếu tìm được quadrilateral), binary, ký tự, overlay box và JSON/CSV prediction. Trạng thái `ocr_prediction_only` nghĩa là mô hình đưa ra chuỗi, không khẳng định chuỗi chính xác. Ảnh hiện có trong workspace **không có ground truth**, nên kết quả debug hiện tại không phải accuracy report. `dataset/evaluation_ground_truth_template.csv` cần người xác minh; hai ví dụ từ ảnh người dùng được đánh dấu `manual_estimate_from_user_image`, không được evaluator tính cho đến khi đổi thành `human_reviewed` sau khi xác minh.

Sau khi gắn nhãn bằng người:

```powershell
python evaluate_labeled.py dataset/evaluation_ground_truth_template.csv --model cnn --output evaluation_results.json
```

Evaluator bỏ qua các hàng chưa đánh dấu `human_reviewed`, ghi exact match/CER/thời gian cho hàng được duyệt. Đánh giá detection IoU/precision/recall cần box ground truth được xác minh riêng.

Evaluator ghi `null` khi chưa có mẫu được duyệt (không coi đó là accuracy/CER bằng 0).
CER toàn tập được tính bằng tổng edit distance chia tổng số ký tự nhãn; UNKNOWN
được tính như dự đoán rỗng và ảnh không đọc được vẫn là mẫu thất bại. Không tự
loại mẫu lỗi để làm đẹp chỉ số. Danh sách kết quả từng ảnh nằm trong `results`.

Debug JSON hiện chứa `production_pipeline`: box, chuỗi raw/post, confidence và
vùng ký tự thực sự dùng bởi `end_to_end_pipeline`. Ảnh `08_production_*` và thư
mục `production_characters` ghi đúng nhánh này. Các candidate rectification và
kết quả cấp trên được đánh dấu `experimental_geometry_not_production`; không
coi đó là kết quả của ứng dụng khi hai nhánh chọn crop khác nhau.

Chạy thử nghiệm hình học đầu vào mà **không thay trọng số/production preprocessing**:

```powershell
python diagnose_recognition.py
```

Kết quả `debug_outputs/geometry_diagnostic.json` chỉ gồm hai ví dụ phát triển.
Thử kéo giãn foreground thay vì square padding chưa nhận đúng hai biển nên
không được bật trong pipeline. Với ảnh 0403, 9 thành phần phân đoạn không có
nghĩa là segmentation đúng: chuỗi nhìn thấy `83FD9972` chỉ có 8 ký tự.

