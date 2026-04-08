# 🏥 ClinRec

<div align="center">

**Patient State Encoding + Self-History Selection + Safe Medication Recommendation**  
**Core pipeline on MIMIC-IV**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-00897B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Hệ thống khuyến nghị thuốc an toàn từ dữ liệu EHR nhiều lần khám trên MIMIC-IV**

*Data Mining + Recommender Systems · MIMIC-IV Dataset*

[Tổng quan](#1-tổng-quan) · [Kiến trúc](#4-kiến-trúc-hệ-thống) · [Cài đặt](#8-cài-đặt-môi-trường) · [Huấn luyện](#10-huấn-luyện)

</div>

---

## 1. Tổng quan

**ClinRec** là pipeline nghiên cứu end-to-end cho bài toán **safe medication recommendation** từ dữ liệu EHR theo chuỗi visit. Phiên bản hiện tại đã được chốt theo **hướng mới** và chỉ giữ lại phần core của hệ thống:

- **Patient State Encoder**
- **Self-history selection**
- **Fusion giữa current state và self-history**
- **Medication prediction**
- **DDI-aware objective**

Hệ thống tập trung trả lời câu hỏi chính:

**"Nên khuyến nghị thuốc gì cho visit hiện tại, dựa trên trạng thái hiện tại và lịch sử quan trọng của chính bệnh nhân, đồng thời hạn chế tương tác thuốc nguy hiểm?"**

> ⚠️ **Lưu ý:** Dự án phục vụ mục đích nghiên cứu và học thuật. Đây không phải là hệ thống triển khai lâm sàng thực tế và không thay thế quyết định chuyên môn của bác sĩ.

---

## 2. Bài toán

Cho một bệnh nhân tại thời điểm hiện tại với dữ liệu lâm sàng và lịch sử trước đó:

- diagnosis codes
- procedure codes
- lab values
- vital signs
- medication history

mục tiêu của hệ thống là dự đoán:

- **tập thuốc phù hợp** cho visit hiện tại,
- đồng thời **giảm nguy cơ drug-drug interaction (DDI)** trong đầu ra.

Bài toán được mô hình hóa như **multi-label medication recommendation** trên chuỗi visit EHR.

---

## 3. Đóng góp chính

- Mã hóa **trạng thái bệnh nhân theo chuỗi visit** bằng encoder thời gian.
- Chỉ giữ lại **self-history selection** trên lịch sử của chính bệnh nhân.
- Hợp nhất **current state** và **self-history summary** để tạo context vector cho dự đoán thuốc.
- Tối ưu **drug recommendation đa nhãn có kiểm soát DDI**.
- Tổ chức pipeline gọn hơn, dễ train hơn và phù hợp hơn với điều kiện chạy local.

---

## 4. Kiến trúc hệ thống

### 4.1. Pipeline mức cao

```text
Input (diag, proc, lab, vital, med_history)
  → PatientStateEncoder
  → Self-history selection
  → Fusion
  → Medication prediction
  → DDI-aware loss
  → Output: drug_logits, drug_probs
```

### 4.2. Diễn giải từng khối

1. **PatientStateEncoder**  
   Nhận đầu vào gồm `diag_codes`, `proc_codes`, `lab_values`, `vital_values`, `med_history`, `visit_mask` và sinh ra:
   - `visit_repr`
   - `state_sequence`
   - `pooled_state`
   - `visit_mask`

2. **Self-history selection**  
   Chọn các visit quan trọng từ **lịch sử của chính bệnh nhân**, thường bằng visit-level attention trên `state_sequence`.

3. **Fusion**  
   Hợp nhất:
   - `current_state`
   - `self_history_summary`

   để tạo `context_vector`.

4. **Medication prediction**  
   Dự đoán:
   - `drug_logits`
   - `drug_probs`

5. **DDI-aware loss**  
   Tối ưu:

   ```text
   total_loss = prediction_loss + lambda_ddi * ddi_loss
   ```

### 4.3. Ký hiệu chính

- `visit_repr`: biểu diễn của từng visit
- `state_sequence`: chuỗi trạng thái bệnh nhân theo thời gian
- `pooled_state`: biểu diễn gộp toàn trajectory
- `current_state`: trạng thái hiện tại dùng để dự đoán
- `self_history_summary`: tóm tắt lịch sử quan trọng của chính bệnh nhân
- `context_vector`: vector hợp nhất cuối cùng
- `drug_logits`, `drug_probs`: đầu ra dự đoán thuốc


## 5. Dataset

Dự án sử dụng **MIMIC-IV** từ PhysioNet.

- Dataset page: https://physionet.org/content/mimiciv/
- Để truy cập cần tài khoản PhysioNet và hoàn thành training bắt buộc của PhysioNet.

### Các bảng thường dùng

- `patients`
- `admissions`
- `icustays`
- `transfers`
- `diagnoses_icd`
- `procedures_icd`
- `labevents`
- `chartevents`
- `prescriptions`
- `emar`, `emar_detail`, `pharmacy`

> Dữ liệu gốc **không commit lên git**. Chỉ lưu trong `data/raw/`.

### Dữ liệu ngoài MIMIC-IV để build DDI matrix

Ngoài dữ liệu gốc MIMIC-IV, pipeline còn cần thêm các file mapping/pharmacology để tạo ma trận tương tác thuốc (DDI):

- `drug-atc.csv`: tệp ánh xạ mã thuốc sang mã ATC.
- `ndc2RXCUI.txt`: tệp ánh xạ NDC sang RXCUI.
- `drugbank_drugs_info.csv`: bảng thông tin thuốc được tải từ DrugBank, dùng để ánh xạ tên thuốc sang chuỗi SMILES của thuốc. File này hữu ích cho các baseline/phần mở rộng liên quan tới biểu diễn phân tử, nhưng **không bắt buộc** nếu chỉ build ma trận DDI nhị phân cơ bản.
- `drug-DDI.csv`: tệp lớn chứa thông tin về tương tác thuốc (DDI), được mã hóa bằng CID/STITCH.
- `RXCUI2atc4.csv`: tệp ánh xạ NDC-RXCUI-ATC4; trong pipeline hiện tại chỉ dùng phần ánh xạ **RXCUI → ATC4**.

### Nguồn tải các file ngoài
- `drug-atc.csv`,`ndc2RXCUI.txt`: https://github.com/kybinn/DrugDoctor/tree/main
- `drug-DDI.csv`: https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
- `drugbank_drugs_info.csv`: https://drive.google.com/file/d/1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk/view?usp=sharing
- `RXCUI2atc4.csv`: lấy từ repo GAMENet, file gốc có tên `ndc2atc_level4.csv`: https://github.com/sjy1203/GAMENet

### Luồng tạo DDI matrix

MIMIC-IV **không cung cấp sẵn DDI matrix**. Ma trận DDI của project được build theo chuỗi ánh xạ sau:

```text
prescriptions.csv.gz (cột ndc)
→ ndc2RXCUI.txt
→ RXCUI2atc4.csv
→ drug-atc.csv
→ drug-DDI.csv
→ drug_ddi.pt / drug_ddi_report.json
```

Trong đó:

- `prescriptions.csv.gz` cung cấp thuốc kê đơn từ MIMIC-IV.
- `ndc2RXCUI.txt` nối NDC trong MIMIC-IV sang RXCUI.
- `RXCUI2atc4.csv` nối RXCUI sang ATC4.
- `drug-atc.csv` nối CID/STITCH với ATC.
- `drug-DDI.csv` cung cấp các cặp thuốc có tương tác ở mức CID/STITCH.

---

## 6. Cấu trúc thư mục

```text
clinrec/
├── data/
│   ├── raw/
│   │   └── hosp/
│   │   └── icu/
│   ├── interim/
│   │   ├── cohort/
│   │   ├── trajectories/
│   │   └── vocab/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── ddi/
│   └── artifacts/
│       └── encoder/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── eval.yaml
├── src/
│   ├── data/
│   │   ├── load_mimic.py
│   │   ├── build_cohort.py
│   │   ├── build_trajectories.py
│   │   ├── build_vocab.py
│   │   ├── build_ddi_matrix.py
│   │   └── dataset.py
│   ├── features/
│   │   ├── diagnosis_encoder.py
│   │   ├── procedure_encoder.py
│   │   ├── lab_processor.py
│   │   ├── vital_processor.py
│   │   └── medication_history.py
│   ├── models/
│   │   ├── patient_state_encoder.py
│   │   ├── history_selector.py
│   │   ├── fusion.py
│   │   ├── medication_decoder.py
│   │   ├── ddi_regularization.py
│   │   └── full_model.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── train_core.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate_core.py
│   │   ├── evaluate_safety.py
│   │   └── evaluate_ablation.py
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       ├── io.py
│       └── device.py
├── notebooks/
│   ├── 01_eda_mimic_iv.ipynb
│   ├── 02_build_cohort.ipynb
│   ├── 03_train_base.ipynb
│   └── 05_train_full_core.ipynb
├── scripts/
│   ├── preprocess.ps1
│   ├── train_core.ps1
│   └── evaluate.ps1
├── tests/
│   ├── test_data.py
│   ├── test_encoder.py
│   ├── test_history_selector.py
│   ├── test_fusion.py
│   └── test_decoder.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

### Ghi chú

- Nếu repo hiện tại vẫn có `src/retrieval/`, `src/graph/`, `src/explainability/` hoặc các notebook/train script của hướng cũ, có thể giữ tạm trong repo nhưng **không dùng trong pipeline mới**.
- Khi dọn repo sạch hơn, nên chuyển các phần đó sang nhánh khác hoặc đánh dấu deprecated.

---

## 7. Luồng gọi chính giữa các module

### 7.1. Pha dữ liệu

```text
load_mimic.py
  → build_cohort.py
  → build_vocab.py / build_ddi_matrix.py
  → build_trajectories.py
  → dataset.py
```

### 7.2. Pha core model

```text
patient_state_encoder.py
  → history_selector.py
  → fusion.py
  → medication_decoder.py
  → ddi_regularization.py
  → full_model.py
```

### 7.3. Pha train

```text
losses.py
  → trainer.py
  → train_core.py
```

### 7.4. Pha đánh giá

```text
metrics.py
  → evaluate_core.py
  → evaluate_safety.py / evaluate_ablation.py
```

---

## 8. Cài đặt môi trường

### Yêu cầu tối thiểu

- Python 3.9+
- Khuyến nghị dùng môi trường ảo
- RAM đủ lớn để xử lý cohort và trajectory từ MIMIC-IV
- GPU là tùy chọn nhưng hữu ích khi train

### Clone repo

```bash
git clone https://github.com/your-username/clinrec.git
cd clinrec
```

### Tạo môi trường ảo

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Cài dependencies

```bash
pip install -r requirements.txt
```

---

## 9. Chuẩn bị dữ liệu

### Bước 1. Đặt dữ liệu MIMIC-IV vào đúng thư mục

Tối thiểu cần bảng thuốc sau:

```text
data/raw/hosp/prescriptions.csv.gz
```

Ngoài ra các bước preprocess cohort / trajectory có thể dùng thêm các bảng khác trong `hosp/` và `icu/`.

### Bước 2. Đặt các file ngoài để build DDI matrix

Đặt các file sau vào thư mục `data/processed/ddi/` hoặc sửa lại path tương ứng trong config:

```text
data/processed/ddi/drug-DDI.csv
data/processed/ddi/drug-atc.csv
data/processed/ddi/ndc2RXCUI.txt
data/processed/ddi/RXCUI2atc4.csv
```

Nếu bạn dùng thêm baseline/phần mở rộng liên quan molecular graph, có thể đặt thêm:

```text
data/processed/ddi/drugbank_drugs_info.csv
```

### Bước 3. Kiểm tra cấu hình

Chỉnh các file sau cho phù hợp với máy của bạn:

- `configs/data.yaml`
- `configs/model.yaml`
- `configs/train.yaml`
- `configs/eval.yaml`

Riêng `configs/data.yaml`, cần kiểm tra các path cho bước build DDI, ví dụ:

```yaml
paths:
  ddi_source_path: data/processed/ddi/drug-DDI.csv
  mimic_prescriptions_path: data/raw/hosp/prescriptions.csv.gz
  ndc_to_rxcui_path: data/processed/ddi/ndc2RXCUI.txt
  rxcui_to_atc4_path: data/processed/ddi/RXCUI2atc4.csv
  drug_atc_path: data/processed/ddi/drug-atc.csv
```

### Bước 4. Chạy preprocessing

**PowerShell**

```powershell
./scripts/preprocess.ps1
```

Hoặc chạy từng bước bằng Python:

```bash
python -m src.data.build_cohort
python -m src.data.build_vocab
python -m src.data.build_ddi_matrix
python -m src.data.build_trajectories
```

### Bước 5. Kiểm tra build DDI thành công

Sau khi chạy `build_ddi_matrix`, cần có:

```text
data/processed/ddi/drug_ddi.pt
data/processed/ddi/drug_ddi_report.json
```

Mở `drug_ddi_report.json` để kiểm tra:

- `matched_pairs > 0`
- `matrix_shape` khớp với drug vocab
- không còn trạng thái `fallback_zero`

Nếu `matched_pairs = 0`, nghĩa là pipeline DDI chưa nối được đúng giữa:

- thuốc trong MIMIC-IV (`prescriptions.csv.gz`)
- file mapping NDC/RXCUI/ATC
- file `drug-DDI.csv`

Sau bước này, các thư mục quan trọng cần xuất hiện:

- `data/interim/cohort/`
- `data/interim/trajectories/`
- `data/interim/vocab/`
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`
- `data/processed/ddi/`

---

## 10. Huấn luyện

### 10.1. Train bản core

```powershell
./scripts/train_core.ps1
```

hoặc:

```bash
python -m src.training.train_core
```

### 10.2. Công thức loss

```text
total_loss = prediction_loss + lambda_ddi * ddi_loss
```

Trong đó:

- `prediction_loss`: BCEWithLogitsLoss cho bài toán multi-label medication recommendation
- `ddi_loss`: regularization dựa trên ma trận DDI
- `lambda_ddi`: hệ số cân bằng giữa accuracy và safety

### 10.3. Output cần theo dõi khi train

- `prediction_loss`
- `ddi_loss`
- `total_loss`
- Jaccard
- F1
- PRAUC
- DDI Rate

---

## 11. Đánh giá

### Các metric chính

- Jaccard
- F1 Score
- PRAUC
- DDI Rate
- Avg #Drugs

### Chạy đánh giá

```powershell
./scripts/evaluate.ps1
```

hoặc:

```bash
python -m src.evaluation.evaluate_core
python -m src.evaluation.evaluate_safety
python -m src.evaluation.evaluate_ablation
```

### Gợi ý ablation đúng hướng mới

- Base encoder + decoder
- + Self-history selection
- + Fusion
- + DDI-aware loss
- Full core

---

## 12. Quy trình xây dựng khuyến nghị

### Pha 1 — Khóa dữ liệu và tiền xử lý

- `load_mimic.py`
- `build_cohort.py`
- `build_vocab.py`
- `build_ddi_matrix.py`
- `build_trajectories.py`
- `dataset.py`

**Điều kiện đạt:** cohort sạch, vocab ổn định, DDI matrix đúng kích thước, batch đầu tiên load được.

### Pha 2 — Dựng encoder

- `patient_state_encoder.py`

**Điều kiện đạt:** forward pass ổn định, sinh được:
- `visit_repr`
- `state_sequence`
- `pooled_state`
- `visit_mask`

### Pha 3 — Thêm self-history selection

- `history_selector.py`

**Điều kiện đạt:** chọn được visit quan trọng từ chính lịch sử bệnh nhân, attention mask đúng, không dùng neighbor branch.

### Pha 4 — Thêm fusion và decoder

- `fusion.py`
- `medication_decoder.py`
- `full_model.py`

**Điều kiện đạt:** full forward pass chạy end-to-end từ batch đến `drug_logits`.

### Pha 5 — Huấn luyện và đánh giá

- `losses.py`
- `trainer.py`
- `train_core.py`
- `evaluate_core.py`
- `evaluate_safety.py`
- `evaluate_ablation.py`

**Điều kiện đạt:** có checkpoint tốt nhất, bảng metric và safety report.

### Pha 6 — Script hóa và test hóa

- `scripts/*`
- `tests/*`
- `README.md`

**Điều kiện đạt:** người khác clone repo có thể preprocess, train và evaluate bản core.

---

## 13. Kiểm thử

Các test chính:

- `tests/test_data.py`
- `tests/test_encoder.py`
- `tests/test_history_selector.py`
- `tests/test_fusion.py`
- `tests/test_decoder.py`

Khuyến nghị chạy test sớm theo từng pha thay vì để đến cuối.

---

## 14. Baselines và paper liên quan

### Baselines nên biết

- GAMENet
- SafeDrug
- MICRON
- COGNet
- MoleRec
- VITA

### Gợi ý theo module

- **Patient state encoding:** RETAIN, BEHRT, Med-BERT
- **Relevant visit selection:** VITA
- **DDI-aware objective:** SafeDrug, MoleRec

> README hiện tại chỉ mô tả **pipeline core đang dùng**. Một số paper như DAPSNet, RaVSNet, HypeMed hoặc các hướng retrieval / hypergraph vẫn có thể xuất hiện trong phần related work của báo cáo, nhưng không phải là thành phần của code pipeline mới.

---

## 15. Artifact đầu ra

Sau khi train / evaluate, các kết quả được lưu tại:

```text
outputs/
├── checkpoints/
├── logs/
├── predictions/
├── figures/
└── reports/
```

Đây là nơi lưu:

- checkpoint tốt nhất
- log huấn luyện
- prediction export
- biểu đồ loss / metric / ablation
- báo cáo tổng hợp cuối cùng

Riêng artifact của bước build DDI được lưu tại:

```text
data/processed/ddi/drug_ddi.pt
data/processed/ddi/drug_ddi_report.json
```

- `drug_ddi.pt`: ma trận DDI dùng khi train/evaluate
- `drug_ddi_report.json`: báo cáo mapping và số cặp DDI match được

---

## 16. Thành viên nhóm

| Thành viên | Vai trò chính |
|---|---|
| Bùi Đức Đại | Data + Features + Patient State Encoder |
| Đỗ Mạnh Cường | Self-history selection + Integration support |
| Nguyễn Văn Phúc | Fusion + Ablation + Model integration |
| Nguyễn Thế Dương | Decoder + Training + Evaluation + Documentation |

---

## 17. Ghi chú sử dụng repo

- Không commit dữ liệu gốc MIMIC-IV.
- Không commit checkpoint lớn, log tạm, cache và file nhạy cảm.
- Ưu tiên ổn định **core pipeline** trước khi thử nghiệm bất kỳ mở rộng nào.
- Logic production nên nằm trong `src/` và `scripts/`, không để notebook là nơi duy nhất chứa code chính.
- Nếu còn file retrieval / graph / explainability trong repo, hãy xem đó là phần cũ và tránh import vào pipeline mới.

---

## 18. License

MIT License

---

## 19. Citation

Nếu bạn sử dụng repo hoặc ý tưởng từ dự án này cho báo cáo / nghiên cứu, hãy trích dẫn repo và các paper baseline liên quan trong phần tài liệu tham khảo.

---

## 20. Disclaimer

ClinRec là hệ thống nghiên cứu phục vụ học thuật. Mọi đầu ra của hệ thống chỉ mang tính chất hỗ trợ phân tích và không được xem là chỉ định lâm sàng thực tế.
