# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026/11/6

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
-> Record này nói về **một vật thể riêng** trong ảnh. Vì trong ảnh có nhiều vật thể trong ảnh
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
-> Danh sách tên lớp (class list) đi kèm taxonomy (ImageNet-1K)
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
-> Vì có thể trong các tập `taxonomy_name` mà có thể trùng `class_id`, `class_name`và để có thể nhận biết input nào có ranks cao hơn
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
->  Loại bỏ hoàn toàn cảm tính và cung cấp một decision tree với độ ưu tiên cụ thể
6. Vì sao không được lấy `score` của model làm ground truth?  
-> Vì score là ước lượng thống kê, có thể có sai sót. Ground truth phải được xác định độc lập bởi con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
-> Góc trên-trái ở (385.33, 69.24), góc dưới-phải ở (498.92, 348.92), rộng 113.58 và cao 279.68 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
-> Ngưỡng 0.2: 17 vật thể và Ngưỡng 0.6: 6 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
-> Khi hạ ngưỡng thì sẽ làm tăng Recall nhưng giảm Presicion, Review phải sàng lọc, dọn dẹp và hiệu chỉnh thủ công
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
-> Mỗi hộp là một thực thể duy nhất, khi có hai đối tượng chồng lên nhau thì phải có hai hộp riêng biệt
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
->  Có person bị cắt mép vì chỉ có mỗi cánh tay xuất hiện. Em sẽ đưa lên hỏi Lab Coach. Lý do: Vì không chắc chắn xác định bị cắt mép như thế nào (Có thể là chân, tay hoặc một phần người xuất hiện trong ảnh)
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-010 ; `class_name`=bowl ; `score`=0.372406 ; số điểm polygon=`polygon_point_count`=12 ; 3–5 điểm đầu của `polygon_xy`=[[155.0, 170.0], [155.0, 177.0], [156.0, 178.0], [156.0, 179.0], [161.0, 184.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
-> Cho ta biết đường nét chi tiết hình dạng vật thể thay vì chỉ là khung hình chữ nhật bao quanh vật thể
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
-> `instance_id` dùng để định danh các cá thể riêng biệt nhau trong một ảnh tĩnh.
    Nó **không** phải class ID vì Instance Id phân biết các vật thể (kitchen-1, kitchen-2) có cùng class (kitchen)
    Nó **không** phải tracking ID vì Instance Id gắn tạm thời cho một vật thể trong một ảnh tĩnh và có thể bị thay đổi khi cũng là vật thể đó khi xuất hiện ở ảnh khác còn tracking Id gắn xuyên suốt và không bị thay đổi và bắt buộc
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
-> Chỉ gán những phần nhìn thấy, tuyệt đối không vẽ suy đoán để vẽ thêm phần thân vật thể bị che khuất bởi vật thể khác
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
-> Owen và dining table chạm vào nhau. Em sẽ dùng guideline vì guidline đã có sẵn đầy đủ bộ nguyên tắc và đảm bảo tiến độ chỉ dùng escalation trong trường hợp quá khó để phân biệt giữa chúng
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh   |Cấp độ toàn ảnh (image-level label), không chứa tọa độ hình học (không có bounding box hay polygon), định dạng Json với Rank, Score, Class set|Các phương tiện phía sau bị che khuất và mờ do sương mù/khói bụi, rất khó phân định rõ ranh giới phân loại nếu không có quy tắc ưu tiên rõ ràng.|Áp dụng quy tắc đối tượng nổi bật và không tin tưởng tuyệt đối vào kết quả của mô hình  | Kiểm tra thiên kiến tự động và Sàng lọc tập dữ liệu 
| Phát hiện vật thể |Cấp độ từng thực thể vật thể (instance-level bounding box). Mỗi vật thể có một nhãn lớp kèm tọa độ hình học ($x_{min}, y_{min}, x_{max}, y_{max}$ hoặc $x_c, y_c, w, h$). Không chứa trường score  | Bỏ sót vật thẻ do điều chỉnh ngưỡng  | Tuân thủ đúng Cây phân loại, xử lý đối tượng cắt mép, rà soát độc lập  |Kiểm tra độ khít, ranh giới hộp; Xem độ bao phủ và xót các vật thể
| Instance segmentation |Cấp độ điểm ảnh theo từng thực thể riêng biệt (Pixel-level mask / Polygon points). Mỗi vật thể có một instance_id duy nhất trong ảnh (ví dụ: kitchen-001, kitchen-002), kèm nhãn lớp (category_id) và danh sách tọa độ đa giác bám sát viền |Chùm lá/thảo mộc khô treo ở góc trên bên trái bị nhận diện thành potted plant 0.63 (cây cảnh trong chậu), dù không có chậu cây nào  |Không gán potted plant cho thảo mộc treo: Hủy bỏ (delete) nhãn sai do model đề xuất; chuyển sang nhãn khác nếu taxonomy hỗ trợ hoặc đánh dấu bỏ qua.  |Kiểm tra tính độc lập của Instance ID của từng vật thể độc lập  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
-> Không được chia sẽ bài lab ra môi trường khác (đăng tải lên mạng xã hội,.....) ngoài pham vi được cho phép của bài lab
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
-> Báo lên cho Lab Coach để xủ lý

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
