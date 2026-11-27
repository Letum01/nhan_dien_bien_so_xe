# Kiểm định cohort OCR trong Learning Center

Mở <http://localhost:8513/?mode=cohort>. Nếu chưa chạy, mở `OPEN_ALPR_LEARNING_CENTER.bat`, rồi dùng URL trên hoặc chọn **Kiểm định cohort OCR** ở thanh bên. Không cần chạy model hoặc mở một server khác.

1. Nhập bí danh người rà soát. Chọn hàng đợi **Glyph (412)**. So crop gốc và vị trí box với nhãn nguồn, rồi bấm **Đúng**, **Sai** hoặc **Không rõ**. Nếu box không đủ nét để xác nhận, đừng suy đoán ký tự. Có thể ghi domain quan sát, loại lỗi và ghi chú.
2. Chọn **Độ đầy đủ box (122)**. So toàn ảnh với overlay và bấm **Đủ box**, **Thiếu box** hoặc **Không rõ**. Nếu không thiếu nhưng có box thừa/lệch, ghi loại lỗi; “Đủ box” không có nghĩa mọi box đúng. Không cần vẽ lại box nào.
3. Mỗi lần bấm quyết định sẽ lưu ngay và chuyển tới mẫu chưa làm tiếp theo. Dùng **Mã mẫu** để mở lại mẫu đã lưu; lần lưu mới tạo revision, không xóa quyết định trước. Sau reload, app chọn mẫu đầu tiên chưa làm. Bí danh chỉ được giữ trong phiên trình duyệt; sau reload có thể phải nhập lại.
4. Mở **Thống kê và xuất kết quả** để xem bảng theo class nguồn, domain do người review ghi, loại lỗi và stratum. Tải CSV riêng cho từng hàng đợi. Domain quan sát không phải provenance; mục không chọn vẫn là “Chưa xác định”. Bộ đếm quyết định gồm cả “Không rõ”; tỷ lệ sai/thiếu chỉ lấy mẫu đã kết luận. Không coi mẫu bỏ trống hay “Không rõ” là âm tính.

Lưu cục bộ vào `data/alpr_cohort_review.sqlite3` (chỉ tạo khi có lưu), không ghi vào SQLite annotation/feedback hiện tại, manifest huấn luyện hoặc CSV mẫu gốc. Không lưu thêm pixel. Kiểm tra hash hàng đợi và đối chiếu ảnh/nhãn gốc khi xem. Không dùng dự đoán model.

## Review theo thứ tự ưu tiên và theo ảnh

- Mở **Glyph (412) → Ưu tiên → Đến mẫu chưa làm**. Chế độ này sắp xếp lại các mã chưa review, không sửa thành viên cohort hay số hiệu. Ưu tiên ảnh có dấu hiệu biển Việt Nam từ review/quan sát cũ, lớp `D T M N K A B S` hoặc lớp có không quá 2 mẫu thật trong baseline 370, và crop tier B. Các glyph cùng ảnh đi liền nhau; ưu tiên thêm ảnh còn nhiều glyph khi mức ưu tiên tương đương.
- Bật **Review nhiều glyph cùng ảnh** để xem overlay và từng crop trong một lượt. Chọn quyết định riêng từng dòng rồi bấm **Lưu các quyết định đã chọn**. Dòng chưa chọn và glyph đã lưu không bị ghi thêm revision. Muốn sửa quyết định cũ, tắt chế độ theo ảnh và dùng **Mã mẫu**.
- Bảng/CSV glyph có nhóm nghiên cứu riêng. **Vietnamese verified** không được tự gán từ ngoại hình hoặc quyết định đúng. Domain quan sát được giữ tách biệt; sai được xếp `Invalid`, không rõ được xếp `Ambiguous`. Các nhóm là để phân tích, không phải danh sách train.
- [Giao thức pilot 370 + 100/300/500](../configs/alpr_small_transfer_pilot_v1.json) là bản dự thảo riêng; không sửa progressive runner. Chỉ fit sau khi hồ sơ quyền, provenance, nhãn, split và phê duyệt benchmark đạt. Giữ một development cohort chung và báo cáo matched baseline đối với mỗi phương pháp, không so nhầm tác dụng của thay recipe với tác dụng của tăng dữ liệu.

## Chất lượng và độ phủ

Các file ảnh và nhãn của **422 ảnh gốc khác nhau** khớp audit; không ảnh nào trong danh sách cách ly đã ghi nhận. 412 glyph đến từ **338 ảnh** (337 thành phần trùng/gần trùng đã ghi nhận), 122 ảnh detector là 122 ảnh khác nhau, 38 ảnh xuất hiện trong cả hai hàng đợi. Glyph có đủ **30 lớp nguồn**, mỗi lớp có 10–25 mục; phủ ảnh có cờ nhỏ, mờ, sát mép, chồng lấn; có mẫu chart, biển nước ngoài và ảnh nghi thiếu nhãn. Phân bố theo split gốc lần lượt 342/70 glyph và 105/17 ảnh detector (`train`/`val`). Không xác minh được nhóm xe độc lập. Xem `reports/progressive_readiness_review_v1/cohort_integrity.json`.

Đây là cohort kiểm định có chủ đích lấy dư lớp hiếm và ảnh rủi ro, không phải mẫu ngẫu nhiên đại diện cho toàn corpus hay tập benchmark độc lập. Các stratum giao nhau và số glyph không bằng số nguồn độc lập. Không suy rộng tỷ lệ lỗi, không tự duyệt phần chưa kiểm tra.

## Điều kiện còn thiếu trước training

- Có văn bản cho phép huấn luyện học thuật trên đúng archive/phạm vi ảnh đóng góp; quyền đánh giá và phân phối xem riêng.
- Có kết quả review thật và bằng chứng chất lượng phù hợp với các mẫu **được chọn**. Review glyph không xác nhận completeness của biển. YOLO cần xác minh đủ và đúng box theo ảnh, không chỉ quyết định “Đủ box”.
- Có source-group có căn cứ, leakage evidence và manifest train/development/test khóa bằng hash, cùng phê duyệt protocol progressive. Không dùng group pHash làm group xe, không chuyển Pilot/New10/Final Test vào train hay tuning.

Khi đủ điều kiện, dùng preflight và runner hiện có theo [báo cáo readiness](../reports/ALPR_PROGRESSIVE_TRAINING_READINESS.md): CNN → SVM → RF, rồi YOLO Character sau gate riêng. Chế độ review không tạo frozen manifest hoặc tự khởi động training.