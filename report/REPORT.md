# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python 3

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:**
   - `class_id` = `468`
   - `class_name` = `cab`
   - `rank` = `1`
   - `score` = `0.510915`
   - `taxonomy_name` = `ImageNet-1K`

2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**
   - Record này nói về **cả bức ảnh**.
   - **Lý do:** Dựa trên ảnh `classification_top5.png`, biểu đồ cột phân bố điểm số xác suất cho toàn bộ bức ảnh `traffic` chứ không gắn với tọa độ hay vùng biên của vật thể cụ thể nào. Ngoài ra, model sử dụng là `yolo11n-cls.pt` (hậu tố `-cls` là viết tắt của Image Classification – phân loại toàn bộ khung hình).

3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**
   - Danh sách tên lớp đi kèm theo taxonomy / checkpoint (cụ thể ở đây là bộ nhãn ImageNet-1K gồm 1.000 lớp gắn liền với checkpoint huấn luyện sẵn `yolo11n-cls.pt`), model không thể tự nghĩ ra lớp mới.

4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**
   - Cần kết hợp cả 3 thông tin để đảm bảo cả hệ thống máy tính lẫn con người đều hiểu đúng và chuẩn hóa:
     - `class_id`: Định danh duy nhất dạng số (unique key) phục vụ máy tính lưu trữ và lập trình xử lý hiệu quả.
     - `class_name`: Định danh dạng chữ giúp con người đọc hiểu trực tiếp mà không cần tra cứu bảng mã.
     - `taxonomy_name`: Xác định không gian nhãn và quy chuẩn phân loại (tránh nhầm lẫn cùng một tên lớp nhưng thuộc hai bộ taxonomy khác nhau).

5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**
   - Guideline cần quy định rõ nguyên tắc ưu tiên:
     - Ưu tiên đối tượng/chủ thể chiếm diện tích lớn nhất trong khung hình hoặc nằm ở trung tâm / điểm lấy nét chính.
     - Phân định rõ bối cảnh khi nào gán nhãn toàn cảnh (scene-level) và khi nào gán nhãn phương tiện/đối tượng cụ thể.
     - Thiết lập thứ tự ưu tiên (hierarchy/precedence) giữa các lớp khi chúng xuất hiện đồng thời trong một bức ảnh.

6. **Vì sao không được lấy `score` của model làm ground truth?**
   - Vì `score` chỉ là xác suất dự đoán mang tính chủ quan của mô hình toán học và hoàn toàn có thể bị sai. Ground truth bắt buộc phải là sự thật khách quan được kiểm chứng và xác nhận bởi con người dựa trên quy tắc nhất quán. Nếu lấy dự đoán của mô hình làm ground truth, ta sẽ biến lỗi sai của máy thành đáp án chuẩn, làm mất đi giá trị đánh giá và kiểm thử mô hình.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record:**
   - `class_name` = `person`
   - `score` = `0.912625`
   - `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]`
   - `bbox_width` = `113.58`
   - `bbox_height` = `279.68`

2. **Giải thích box bằng lời thường:**
   - **Góc trên - trái:** Tọa độ $(x_{min}, y_{min}) = (385.33, 69.24)\text{ px}$ — cách mép trái ảnh khoảng $385\text{ px}$ và cách mép trên ảnh khoảng $69\text{ px}$.
   - **Góc dưới - phải:** Tọa độ $(x_{max}, y_{max}) = (498.92, 348.92)\text{ px}$ — cách mép trái ảnh khoảng $499\text{ px}$ và cách mép trên ảnh khoảng $349\text{ px}$.
   - **Kích thước hộp:** Chiều rộng là $113.58\text{ px}$ ($498.92 - 385.33$), chiều cao là $279.68\text{ px}$ ($348.92 - 69.24$).

3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
   - Ngưỡng 1 (`0.20`): `17` predictions
   - Ngưỡng 2 (`0.60`): `6` predictions

4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**
   - Rà soát và xóa bỏ các hộp dự đoán sai (loại bỏ False Positives).
   - Xóa bỏ các hộp bị phát hiện trùng lặp trên cùng một vật thể.
   - Nắn chỉnh lại tọa độ các cạnh của box để ôm khít chính xác đối tượng.

5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):**
   - Hộp giới hạn (bounding box) phải bao trọn toàn bộ các phần nhìn thấy được của vật thể; bốn cạnh của hộp phải tiếp xúc sát nhất có thể với mép ngoài cùng của đối tượng, không chừa khoảng trống nền thừa thãi và tuyệt đối không cắt lẹm vào thân vật thể.

6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
   - **Vật bị che khuất:** Lò nướng phía sau (`oven` 0.69) bị chiếc bàn làm việc phía trước che mất phần thân dưới.
   - **Phương án:** Viết thành quy tắc trong **guideline** (không cần escalation).
   - **Vì sao:** Che khuất cục bộ (occlusion) và cắt mép (truncation) là các tình huống phổ biến trong thị giác máy tính. Việc chuẩn hóa trực tiếp quy tắc này vào guideline (ví dụ: _"vẫn đóng box bao quanh phần nhìn thấy nếu nhận diện rõ >30% diện tích"_) sẽ giúp người gán nhãn chủ động thao tác nhất quán mà không gây gián đoạn quy trình.

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record:**
   - `instance_id` = `kitchen-001`
   - `class_name` = `person`
   - `score` = `0.899318`
   - Số điểm polygon (`polygon_point_count`) = `348`
   - 3–5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`

2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
   - Polygon cung cấp chi tiết đường bao thực tế (silhouette) bám sát từng đường cong, nếp uốn của đối tượng thay vì chỉ là khung chữ nhật thô bao quanh.
   - Thể hiện chính xác tư thế, phần rỗng và cấu trúc hình học phức tạp (ví dụ: khoảng trống giữa hai chân hoặc giữa cánh tay và thân mình).

3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
   - `instance_id` là mã định danh cục bộ nhằm phân biệt riêng rẽ từng cá thể vật thể độc lập trong cùng một bức ảnh (ví dụ: phân biệt cái bát thứ nhất `kitchen-002`, cái bát thứ hai `kitchen-003`, cái bát thứ ba `kitchen-006` dù chúng cùng có `class_name = bowl`).
   - Nó không phải `class_id` vì không đại diện cho danh mục loài/lớp. Nó không phải `tracking_id` vì chỉ tồn tại trong ảnh tĩnh này, không có giá trị duy trì theo dõi qua các frame video kế tiếp.

4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo:**
   - Người gán nhãn phải chấm các đỉnh polygon ôm khít theo mép ngoài phần nhìn thấy được của vật thể ở mức độ điểm ảnh (pixel). Không vẽ lem lấn ra vùng nền và không cắt lẹm vào chi tiết vật thể. Nếu vật thể có lỗ thủng hoặc khoảng trống xuyên thấu (như quai nồi, kẽ hở tay), phải loại trừ vùng rỗng đó ra khỏi mask.

5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
   - **Vị trí:** Chiếc âu sứ trắng (`bowl` 0.74) đặt tiếp xúc trực tiếp trên mặt bàn gỗ (`dining table` 0.58), phần đáy âu che đè lên diện tích mặt bàn.
   - **Phương án:** Dùng **guideline**.
   - **Vì sao:** Đây là trường hợp quan hệ không gian cơ bản giữa vật thể và bề mặt đỡ. Quy định sẵn trong guideline quy tắc phân tách ranh giới và thứ tự layer tiếp xúc sẽ giúp toàn bộ đội ngũ gán nhãn xử lý đồng bộ, tránh tắc nghẽn tiến độ do phải chờ giải đáp từng trường hợp.

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                    | Ground truth trông như thế nào? (đơn vị nhãn)                                             | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                     | Annotator làm gì?                                                                                         | Reviewer xem gì?                                                                                                         |
| :------------------------ | :---------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Phân loại ảnh**         | 1 nhãn lớp duy nhất gán cho toàn bộ bức ảnh.                                              | Model đoán top 1 là `cab` (`score = 0.510915`) dù khung cảnh giao thông có cả xe buýt và các phương tiện nổi bật khác.                 | Quan sát toàn bộ bức ảnh, đối chiếu quy tắc ưu tiên trong guideline để chọn đúng 1 nhãn đại diện cho ảnh. | Kiểm tra nhãn được gán có đúng quy tắc ưu tiên trong guideline không (đã chọn đúng chủ thể chính chưa).                  |
| **Phát hiện vật thể**     | Bounding box $(x_{min}, y_{min}, x_{max}, y_{max})$ kèm nhãn lớp cho từng cá thể vật thể. | Xử lý vật thể bị che khuất một phần (`oven` 0.69 bị bàn che) hoặc bị cắt ở mép ảnh (`bowl` 0.72).                                      | Kéo vẽ bounding box ôm khít từng vật thể nhìn thấy và gán đúng nhãn lớp tương ứng.                        | Soát độ khít 4 cạnh box, kiểm tra vật bị bỏ sót (False Negative), box vẽ nhầm nền (False Positive) và gán sai nhãn.      |
| **Instance segmentation** | Đa giác khép kín (`polygon_xy`) kèm `instance_id` và nhãn lớp cho từng cá thể.            | Hai vật thể chạm sát / che đè nhau (đáy `bowl` 0.74 đè lên mặt `dining table` 0.58), biên vùng bóng tối khó phân định ranh giới pixel. | Chấm các điểm tọa độ tạo polygon ôm sát mép đường viền (silhouette) thực tế của từng cá thể.              | Soát độ chính xác từng pixel của đường viền (không lem, không lẹm), kiểm tra vùng hõm/rỗng và chỗ tiếp xúc giữa các vật. |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)**
   - Chỉ thao tác và xử lý dữ liệu trực tiếp trong môi trường thực hành được cấp phép (máy ảo Google Colab/hệ thống bài lab), tuyệt đối không tải về máy cá nhân, không phát tán hay chia sẻ dữ liệu ra bên ngoài.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**
   - Cần lập tức dừng thao tác trên mẫu dữ liệu đó và báo cáo trực tiếp cho **Lab Coach** (hoặc Giảng viên / Trợ giảng phụ trách) để thực hiện quy trình xử lý ngoại lệ (escalation).

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
