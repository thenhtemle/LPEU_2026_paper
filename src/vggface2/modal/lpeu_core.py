"""LPEU core — ghép từ các cell của notebook lpeu-mufac-person-resubmit (Cell A..K + F2, H/I bản resubmit).
Không sửa logic; chỉ dùng làm thư viện cho vgg_pipeline.py."""
import os, math, json, copy, random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from PIL import Image

# ===========================================================================
# ▓  CELL B: CONFIG
# ===========================================================================
 
@dataclass
class Config:
    # ── paths ─────────────────────────────────────────────────────────────
    dataset_root:  str = "/kaggle/input/datasets/thenhtemle/custom-korean-family-faces"
    # v_person_B — HƯỚNG B (chủ động giảm forget-accuracy, kiểu NegGrad/GLI):
    # output_dir RIÊNG so với file gốc và file Hướng A. split_mufac_by_person
    # KHÔNG đổi (cùng seed) và original-model training (Cell E) không phụ
    # thuộc forget loss/w_kl_uniform — CÓ THỂ tái sử dụng original_model.pt
    # đã train từ lpeu-v7-mufac-person.ipynb (hoặc lpeu-v7-mufac.ipynb) để
    # tiết kiệm ~1-2 giờ train model gốc:
    #   import shutil, os
    #   os.makedirs("./runs/lpeu_v7_mufac_person_B", exist_ok=True)
    #   shutil.copy("./runs/lpeu_v7_mufac_person/original_model.pt",
    #               "./runs/lpeu_v7_mufac_person_B/original_model.pt")
    # (chạy TRƯỚC Cell L)
    output_dir:    str = "./runs/lpeu_v7_mufac_person_B"
    seed:          int = 42
    device:        str = "cuda" if torch.cuda.is_available() else "cpu"
 
    # ── data ──────────────────────────────────────────────────────────────
    image_size:               int   = 128
    batch_size:               int   = 32
    num_workers:              int   = 2
    min_images_per_identity:  int   = 8
    train_ratio:              float = 0.70
    val_ratio:                float = 0.10
    test_ratio:               float = 0.20
    # v7.7 — tỉ lệ hộ gia đình bị giữ lại HOÀN TOÀN ngoài train/val/test,
    # chưa từng được model gốc nhìn thấy. Cần thiết để tính đúng Forgetting
    # Score / MIA kiểu Choi & Na (2023) / GLI (Choi et al. 2024) — so
    # forget vs UNSEEN, KHÔNG phải forget vs retain-test (khác bản chất).
    unseen_ratio:             float = 0.10
 
    # ── model ─────────────────────────────────────────────────────────────
    embedding_dim:       int   = 128
    pretrained_backbone: bool  = True
    arcface_s:           float = 32.0
    arcface_m:           float = 0.30
 
    # ── original training ─────────────────────────────────────────────────
    original_epochs:      int   = 120
    original_lr:          float = 0.05
    original_momentum:    float = 0.9
    original_wd:          float = 1e-4
    warmup_epochs:        int   = 5
    early_stop_patience:  int   = 20   # dừng nếu val_acc không cải thiện sau N epoch
    lr_reduce_patience:   int   = 4    # giảm LR mỗi N epoch không cải thiện
 
    # ── dataset splits ────────────────────────────────────────────────────
    # v_person — FILE NÀY: num_forget_ids nghĩa là số CÁ NHÂN (person_id) bị
    # quên, KHÔNG phải số hộ gia đình (khác file lpeu-v7-mufac.ipynb gốc,
    # nơi num_forget_ids = số HOUSEHOLD). Xem split_mufac_by_person (Cell C).
    # v_scaleup (2026-08-17) — tăng từ 20 → 100 người (809 người đủ điều
    # kiện trong tập train, ~12% dân số) để: (1) tăng mia_n_per_class (155 →
    # ~775 ước tính), giảm std bootstrap gần một nửa, đủ phân giải để thấy
    # tín hiệu thật nếu có; (2) tăng số ảnh forget mỗi epoch, tăng cơ hội
    # L_kl_uni (file B) đẩy embedding vượt qua ranh giới quyết định thay vì
    # chỉ làm mềm entropy mà không lật dự đoán.
    num_forget_ids: int = 100
    neighbor_k:     int = 64
 
    # ── staged unlearning schedule ─────────────────────────────────────────
    # v7.4-tune: log MUFAC thực tế cho thấy proto_sim gần như không
    # nhích (drop +0.0000) NGAY CẢ ở epoch 1 (anchor tắt hoàn toàn,
    # không gì cản trở) — nghi ngờ 1 epoch ERASE thuần là quá ngắn để
    # thấy hiệu quả trước khi anchor can thiệp từ epoch kế tiếp. Tăng
    # lên 3 (giữ tổng 20 epoch bằng cách giảm stage2 tương ứng).
    stage1_epochs: int = 3    # Phase 1: anchor OFF
    # v7.4-tune: log MUFAC thực tế cho thấy proto_sim GIẢM ĐỀU, gần
    # tuyến tính suốt 20 epoch, KHÔNG hề chững lại ở cuối (Phase 2 còn
    # giảm nhanh hơn Phase 1) — dấu hiệu cơ chế đúng hướng nhưng cần
    # nhiều thời gian hơn ngân sách 20 epoch hiện tại. Kéo dài Phase
    # 2/3 để kiểm chứng xu hướng có tiếp tục hay chững lại.
    stage2_epochs: int = 20   # Phase 2: anchor 50%
    stage3_epochs: int = 8    # Phase 3: anchor 100%
 
    # ── LR ────────────────────────────────────────────────────────────────
    unlearn_lr:      float = 5e-6
    unlearn_wd:      float = 1e-6
    layer4_lr_mult:  float = 0.01
 
    # Manual step size cho bước FORGET (neck + layer4) — bypass AdamW vì
    # nhân gradient với hằng số boost gần như vô tác dụng khi đi qua AdamW
    # (m/√v tự chuẩn hoá scale). Xem chi tiết trong lpeu_v7.py.
    # v7.4-tune: log MUFAC thực tế — erase/repel loss giữ nguyên mức
    # cao (~0.84-0.87 / ~0.98-1.05) suốt 20 epoch mà proto_sim gần như
    # đứng yên (1.0000→0.9992) → nghi ngờ step size quá nhỏ so với
    # thang gradient thực tế của task 8-lớp tuổi (khác VGGFace2/Korean
    # Family). Tăng ~6-7× để kiểm chứng giả thuyết này.
    forget_manual_lr: float = 1e-3
 
    # ── retain path ───────────────────────────────────────────────────────
    w_kd_global:    float = 30.0
    kd_temperature: float = 2.0
    k_anchor:       int   = 30
    w_kd_local:     float = 15.0
    use_ewc:        bool  = True
    w_ewc:          float = 50.0
    w_ce_retain:    float = 5.0
 
    # ── RETAIN-PROTOTYPE ANCHOR ──────────────────────────────────────────
    use_proto_retain: bool  = True
    w_proto_retain:   float = 20.0
 
    # ── IN-LOOP REPAIR ────────────────────────────────────────────────────
    use_inloop_repair: bool  = True
    w_repair:          float = 20.0
 
    # ── POST-UNLEARNING REPAIR (2-phase) ────────────────────────────────
    use_post_repair:  bool  = True
    # v7.4-tune: đã thử tăng repair_epochs/repair_lr để đẩy retain_accuracy
    # — kết quả thực tế: retain tăng (54.2%) nhưng forget_sim_drop/
    # cluster_sim_drop rơi về ÂM (repair không có gì bảo vệ phần đã xoá,
    # CE+KD kéo neck/layer4 trôi ngược về gần cấu hình gốc). REVERT lại
    # giá trị đã cho kết quả cân bằng tốt nhất, chuyển sang bảo vệ bằng
    # repair_forget_guard_weight thay vì đẩy repair mạnh hơn.
    repair_epochs:    int   = 20
    repair_lr:        float = 5e-5
    repair_layer4_lr_mult: float = 0.35   # tuned riêng cho MUFAC (family khó
                                           # tách hơn VGGFace2 — xem lpeu_v7.py)
    repair_r2_kd_mult: float = 0.15       # tuned riêng cho MUFAC
 
    # v7.4: GUARD MỚI — repair phase (chỉ tối ưu CE+KD trên retain) không
    # có gì ngăn neck/layer4 trôi ngược về gần cấu hình gốc, vô tình xoá
    # luôn phần forget vừa xoá được (quan sát thực tế: đẩy repair mạnh
    # hơn → forget_sim_drop/cluster_sim_drop rơi về âm). Thêm 1 lực nhỏ,
    # song song với CE/KD, tiếp tục đẩy embedding forget ra xa
    # old_forget_proto (mirror L_erase của ForgetLossV7 nhưng nhẹ hơn
    # nhiều — không có PCGrad/boost/manual-SGD, chỉ AdamW thường ở
    # repair_lr) — mục tiêu: cho phép CE hồi phục retain_accuracy mà
    # KHÔNG kéo cụm forget trôi ngược lại. Giá trị cần kiểm chứng thực
    # nghiệm — 0.0 = tắt hẳn (quay về hành vi cũ).
    # v7.4-tune: guard_weight=3.0 chỉnh quá tay — retain_accuracy rơi
    # xuống 49.0% (thấp nhất trong 4 phương pháp), thấp hơn cả lần KHÔNG
    # có guard (53.3%) dù cùng repair_epochs/repair_lr. Hạ xuống còn một
    # nửa để tìm điểm cân bằng nhẹ hơn giữa CE-recovery và chống un-erase.
    repair_forget_guard_weight: float = 1.5
 
    # v7.4: PHASE R3 CATCH-UP — thay vì trông chờ may rủi giữa các lần
    # chạy, chủ động chạy THÊM epoch CE (vẫn có guard đi kèm) sau R1+R2
    # nếu retain_accuracy chưa bằng/vượt Original, cho tới khi bắt kịp
    # hoặc chạm trần epoch (tránh vòng lặp vô hạn nếu target không thể
    # đạt được — lúc đó chấp nhận kết quả tốt nhất tìm được).
    # v7.6 — mục tiêu hiện tại: retain_accuracy CAO NHẤT, tạm gác các
    # metric/hệ số khác. Bật catch-up (trước đó tắt mặc định), nâng target
    # lên >1.0 để chủ động VƯỢT Original thay vì chỉ hoà, và nới trần epoch.
    # guard_weight (ở trên, repair_forget_guard_weight=1.5) giữ NGUYÊN — đây
    # là giá trị đã tuned an toàn qua thực nghiệm (guard=3.0 quá tay, guard=0
    # gây un-forget) nên chỉ đẩy target/max_epochs, không đụng guard.
    repair_catchup_enable:     bool  = True
    # v7.9 — HẠ 1.03 → 1.00 (đảo ngược v7.6, cùng lý do gate ở trên). Target
    # >1.0 chủ động chạy THÊM epoch repair để VƯỢT retain gốc; guard chống
    # un-erase trong repair (repair_forget_guard_weight=1.5) yếu hơn nhiều so
    # với lực CE+KD+proto kéo retain lên (5.0 + KD + 20.0) — càng nhiều epoch
    # repair, càng nhiều rủi ro kéo cluster forget trôi ngược gần cấu hình
    # gốc. target=1.00: repair dừng khi đã HOÀ retain gốc, không cố vượt
    # thêm — giảm số epoch catch-up thực chạy, giảm rủi ro xói mòn forgetting.
    repair_catchup_target:     float = 1.00
    # v7.7 KHẨN — nới trần 25→30 (embedding MUFAC không collapse nên CE hồi
    # phục hiệu quả hơn VGGFace2, chỉ cần thêm chút room để chắc ăn).
    repair_catchup_max_epochs: int   = 30
 
    # ── forget path ───────────────────────────────────────────────────────
    w_erase:       float = 2.0
    # HƯỚNG B — chủ động giống NegGrad/GLI: dùng L_kl_uni để ép entropy
    # logits forget về uniform (≈ ép forget_accuracy về gần 1/C, giảm hẳn
    # task performance trên ảnh forget — khác Hướng A/task-agnostic). v_B —
    # GIỜ ĐƯỢC STAGE HOÁ theo Migrate→Shatter (w_kl_uni=0 trong MIGRATE, đủ
    # w_kl_uniform trong SHATTER) thay vì chạy full-weight suốt cả quá trình
    # như bản gốc — xem get_forget_subphase_weights (Cell I) và
    # ForgetLossV7.forward (Cell H).
    # v_scaleup (2026-08-17) — tăng 5.0 → 20.0. Lý do: run trước (post
    # checkpoint-fix) cho thấy entropy TĂNG THẬT (+0.020) nhưng không đủ
    # mạnh để lật argmax của bất kỳ ảnh forget nào trong số 155 ảnh (ArcFace
    # head bị đóng băng trong bước forget ở MUFAC — xem ghi chú ở
    # run_lpeu_v7_unlearning). Tăng trọng số này để kiểm tra giả thuyết: cần
    # lực mạnh hơn nữa mới đủ "đẩy" embedding vượt qua ranh giới quyết định
    # cố định, chứ không chỉ làm mềm xác suất.
    w_kl_uniform:  float = 20.0
    w_repel:       float = 4.0
    repel_margin:  float = -0.3

    # ── Hướng C: L_div (PIU-FR-style class-matched KL-to-unseen) ───────────
    # v_C (2026-08-18) — Đọc PIU-FR (Ha et al., đã trích [b4] trong bài đã
    # gửi) + khảo sát Machine Unlearning: L_kl_uni (trên) chỉ ép entropy về
    # UNIFORM — một PROXY, không đảm bảo phân phối logits trên ảnh forget
    # thực sự KHỚP với phân phối trên ảnh CHƯA TỪNG THẤY (x_unseen) — trong
    # khi đó chính xác là điều Forgetting Score (MIA loss-based) đo. L_div =
    # KL(P_orig(x_unseen cùng age-class) ‖ P_forget(x_forget)) — kéo TRỰC
    # TIẾP hành vi của forget-model trên x_forget về đúng hành vi của model
    # GỐC (đóng băng) trên dữ liệu THẬT SỰ chưa từng thấy, cùng phân bố lớp
    # tuổi (class-matched) để so hợp lệ. Đây LÀ mục tiêu MIA kiểm tra, không
    # phải proxy — nhắm thẳng vào Forgetting Score. Stage hoá giống
    # w_kl_uniform: 0 trong MIGRATE (tránh giành lực với L_erase), đủ trọng
    # số trong SHATTER (xem get_forget_subphase_weights, Cell I). Không đụng
    # cơ chế đóng băng ArcFace / retain stack — retain_accuracy không đổi
    # theo cơ chế này.
    w_div:         float = 20.0

    # ── MIGRATE → SHATTER (2-stage forget loss) ─────────────────────────────
    # v7.8 — thay vì chạy L_erase (di chuyển cụm ra khỏi vị trí cũ) và
    # L_repel (phá cụm, rải embedding ra) CÙNG LÚC mỗi step (dễ triệt tiêu
    # nhau: L_repel kéo embedding phủ đều mọi hướng trên mặt cầu, vô tình
    # lấp lại đúng vùng L_erase đang cố dọn trống), tách thành 2 sub-phase
    # NỐI TIẾP trong forget path:
    #   MIGRATE: chỉ L_erase (w_repel=0) — dồn hết lực đẩy cụm forget rời
    #            khỏi prototype cũ.
    #   SHATTER: L_erase hạ xuống một sàn dư (shatter_erase_floor × w_erase,
    #            giữ lực nhỏ chống trôi ngược) + L_repel đầy đủ — phá rã cụm
    #            đã dịch chuyển.
    # Chuyển pha khi proto_drop đạt migrate_target_fraction của mức giảm lý
    # thuyết tối đa (1 + initial_proto_sim), hoặc chạm migrate_max_epochs
    # (an toàn nếu proto_drop không đạt ngưỡng). Chuyển pha 1 CHIỀU — không
    # quay lại MIGRATE sau khi đã sang SHATTER.
    migrate_target_fraction: float = 0.35
    migrate_max_epochs:      int   = 10
    shatter_erase_floor:     float = 0.18
 
    # ── PCGrad ────────────────────────────────────────────────────────────
    use_pcgrad:   bool  = True
    pcgrad_boost: float = 40.0
 
    # ── Gradient projection (v7.5) ───────────────────────────────────────
    # Chiếu gradient của neck.weight (forget step) lên phần bù trực giao của
    # span(local KNN retain-prototypes) — ràng buộc CỨNG ở cấp tham số, đảm
    # bảo bước cập nhật erase KHÔNG THỂ dịch chuyển embedding theo hướng các
    # identity retain gần nhất, bất kể input. Bổ sung cho PCGrad (vốn chỉ xử
    # lý xung đột theo batch hiện tại, nhiễu và không cố định).
    use_grad_projection: bool = True
 
    # ── clipping ──────────────────────────────────────────────────────────
    grad_clip_retain: float = 0.1
    grad_clip_forget: float = 2.0
 
    # ── safety ────────────────────────────────────────────────────────────
    retain_acc_floor: float = 0.90
    # v7.4-tune: log MUFAC thực tế — epoch có proto_drop CAO NHẤT cả
    # run (epoch 19, +0.0139) bị epoch kế tiếp (proto_drop chỉ +0.0128
    # nhưng retain cao hơn) "vượt mặt" làm best-checkpoint, vì công
    # thức balance=forget_score×retain_score cho retain trọng số quá
    # lớn ngay cả khi đã đủ tốt. checkpoint_retain_gate: một khi
    # retain_score vượt ngưỡng này, chỉ còn forget_score quyết định
    # (không thưởng thêm cho retain cao hơn nữa) — xem get_ckpt_balance().
    # v7.6 — nâng gate từ 0.85 → 0.99: một khi retain_score vượt gate, balance
    # chỉ còn phụ thuộc retain_score (không phải forget_score) — luôn ưu tiên
    # tuyệt đối epoch retain cao nhất, đúng mục tiêu hiện tại lúc đó. Gate=0.85
    # (bản gốc) chủ động HY SINH retain sau ngưỡng 85% để đổi lấy forget_score
    # cao hơn.
    # v7.9 — HẠ LẠI 0.99 → 0.85 (đảo ngược v7.6). Mục tiêu đã đổi: giờ cần CẢI
    # THIỆN FORGETTING SCORE, không còn là "retain cao nhất" nữa. Retain-
    # protection (EWC/anchor/in-loop-repair) mạnh nên retain_score thường vượt
    # 99% rất sớm trong vòng lặp — gate=0.99 khiến balance bỏ qua forget_score
    # gần như suốt quá trình, kể cả khi Migrate→Shatter đạt proto_drop lớn ở
    # epoch sau: epoch đó vẫn thua một epoch retain nhỉnh hơn nhưng forget kém
    # hơn nhiều. Bằng chứng thực nghiệm (run MUFAC-person 2026-08-16):
    # cluster_sim_drop cuối cùng chỉ 0.00036 (~0%) dù đã chạy đủ Migrate+
    # Shatter — dấu hiệu best-checkpoint bị chọn gần như thuần theo retain.
    checkpoint_retain_gate: float = 0.85
    safety_grace_epochs: int = 3
    safety_patience:     int = 2
 
    # ── MIA ───────────────────────────────────────────────────────────────
    mia_n_members:    int = 300
    mia_n_nonmembers: int = 300
 
    unlearn_epochs: int = 0   # tính tự động trong __post_init__
 
    def __post_init__(self):
        self.unlearn_epochs = self.stage1_epochs + self.stage2_epochs + self.stage3_epochs
 
 
cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
 
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 
set_seed(cfg.seed)
 
print(f"{'─'*65}")
print(f"  LPEU-MUFAC  |  Device: {cfg.device}")
print(f"{'─'*65}")
print(f"  Model:  emb_dim={cfg.embedding_dim}  s={cfg.arcface_s}  m={cfg.arcface_m}  [Simple Neck]")
print(f"  Staged unlearning: Phase1={cfg.stage1_epochs}e / Phase2={cfg.stage2_epochs}e / Phase3={cfg.stage3_epochs}e")
print(f"  EWC:    {'ON' if cfg.use_ewc else 'OFF'}  w_ewc={cfg.w_ewc}")
print(f"  Forget: w_erase={cfg.w_erase}  w_repel={cfg.w_repel}  margin={cfg.repel_margin}  boost={cfg.pcgrad_boost}×")
print(f"  Forget manual LR (bypass AdamW): {cfg.forget_manual_lr:.1e}")
print(f"{'─'*65}")

# ===========================================================================
# ▓  CELL C: DATASET UTILITIES (MUFAC)
# ===========================================================================

class IndexedSubset(Dataset):
    """Wraps a dataset; returns (x, y, original_idx)."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
    def __len__(self):  return len(self.indices)
    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        return x, y, self.indices[i]


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf


def make_loader(dataset, batch_size, shuffle, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


AGE_CLASS_TO_LABEL = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}
LABEL_TO_AGE_RANGE = {0:"0-6",1:"7-12",2:"13-19",3:"20-30",
                       4:"31-45",5:"46-55",6:"56-66",7:"67-80"}

class MUFACImageDataset(Dataset):
    """.samples=[(path,y)] với y=age_class (0-7). .households lấy trực tiếp
    từ cột family_id trong CSV (không parse từ filename — chuẩn hơn)."""
    def __init__(self, csv_path, image_dir, transform=None):
        meta = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.samples    = [(r["image_path"], AGE_CLASS_TO_LABEL[r["age_class"]])
                            for _, r in meta.iterrows()]
        self.households = [r["family_id"] for _, r in meta.iterrows()]
        self.person_ids = [r["person_id"] for _, r in meta.iterrows()]  # dự phòng

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def load_mufac_dataset(cfg: Config):
    """valid_classes = 8 age-group (0..7), KHÔNG phải identity — num_classes
    downstream tự động thành 8."""
    train_tf, eval_tf = build_transforms(cfg.image_size)
    root = cfg.dataset_root
    csv_path = os.path.join(root, "custom_train_dataset.csv")
    img_dir  = os.path.join(root, "train_images")
    train_base = MUFACImageDataset(csv_path, img_dir, transform=train_tf)
    eval_base  = MUFACImageDataset(csv_path, img_dir, transform=eval_tf)
    filtered_indices = list(range(len(train_base.samples)))
    valid_classes = sorted(set(y for _, y in train_base.samples))
    return train_base, eval_base, filtered_indices, valid_classes


def split_mufac_by_identity(dataset, filtered_indices, cfg: Config):
    """Chia theo family_id (household) — KHÔNG theo age_class. Tự chia
    train/val/test THEO TỪNG HOUSEHOLD ngay trong train_images (không dùng
    custom_val_dataset.csv gốc — xem giải thích ở đầu file).

    v7.7 — THÊM D_unseen: trước khi chia train/val/test, trích riêng một số
    hộ gia đình (unseen_ratio, mặc định 10%) giữ HOÀN TOÀN ngoài mọi giai
    đoạn train — chưa từng được model gốc nhìn thấy. Đây là điều kiện bắt
    buộc để tính đúng Forgetting Score / MIA kiểu Choi & Na (2023) / GLI
    (Choi et al. 2024): hai công trình đó so loss của x_forget với loss của
    x_unseen (KHÔNG phải x_retain_test). Trước bản vá này, compute_mia_auc
    so forget với retain_test — một tín hiệu khác bản chất (đo được "phân
    biệt hộ-bị-quên với hộ-còn-nhớ", không phải "phân biệt hộ-bị-quên với
    dữ-liệu-chưa-từng-huấn-luyện")."""
    household_to_indices: Dict[str, List[int]] = {}
    for idx in filtered_indices:
        hh = dataset.households[idx]
        household_to_indices.setdefault(hh, []).append(idx)

    all_households = sorted(household_to_indices.keys())
    random.seed(cfg.seed)
    shuffled_households = all_households.copy()
    random.shuffle(shuffled_households)
    n_unseen = max(1, int(len(shuffled_households) * cfg.unseen_ratio))
    unseen_households = set(shuffled_households[:n_unseen])
    trainable_households = [hh for hh in all_households if hh not in unseen_households]

    unseen_idx = []
    for hh in unseen_households:
        unseen_idx.extend(household_to_indices[hh])

    train_idx, val_idx, test_idx = [], [], []
    eligible_households = []

    for hh in trainable_households:
        idxs = household_to_indices[hh]
        random.shuffle(idxs)
        n = len(idxs)
        n_tr = max(1, int(n * cfg.train_ratio))
        n_va = max(1, int(n * cfg.val_ratio))
        n_te = max(1, n - n_tr - n_va)
        if n_tr + n_va + n_te > n:
            n_tr = max(1, n_tr - 1)
        train_idx.extend(idxs[:n_tr])
        val_idx.extend(idxs[n_tr:n_tr+n_va])
        test_idx.extend(idxs[n_tr+n_va:n_tr+n_va+n_te])
        if n_tr >= 1 and n_te >= 1:
            eligible_households.append(hh)

    random.seed(cfg.seed)
    forget_households = set(random.sample(eligible_households, cfg.num_forget_ids))

    forget_idx, retain_train_idx = [], []
    for idx in train_idx:
        hh = dataset.households[idx]
        (forget_idx if hh in forget_households else retain_train_idx).append(idx)

    retain_test_idx, forget_test_idx = [], []
    for idx in test_idx:
        hh = dataset.households[idx]
        (forget_test_idx if hh in forget_households else retain_test_idx).append(idx)

    retain_val_idx, forget_val_idx = [], []
    for idx in val_idx:
        hh = dataset.households[idx]
        (forget_val_idx if hh in forget_households else retain_val_idx).append(idx)

    return dict(
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        forget_idx=forget_idx, retain_train_idx=retain_train_idx,
        retain_test_idx=retain_test_idx, forget_test_idx=forget_test_idx,
        retain_val_idx=retain_val_idx, forget_val_idx=forget_val_idx,
        forget_ids=sorted(forget_households),
        unseen_idx=unseen_idx,
        unseen_households=sorted(unseen_households),
    )


def split_mufac_by_person(dataset, filtered_indices, cfg: Config):
    """v_person — BIẾN THỂ của split_mufac_by_identity: forget-unit là CÁ
    NHÂN (person_id) thay vì CẢ HỘ GIA ĐÌNH (family_id).

    QUAN TRỌNG: train_idx / val_idx / test_idx / unseen_idx được tính Y HỆT
    split_mufac_by_identity (cùng cfg.seed, cùng thứ tự random — chỉ khác
    bước CHỌN AI BỊ QUÊN ở cuối). Nhờ vậy original_model.pt đã train bằng
    split_mufac_by_identity DÙNG LẠI ĐƯỢC (xem ghi chú output_dir ở Cell B)
    — không cần train lại model gốc (~1-2 giờ) chỉ để đổi forget-unit.

    Trả thêm `forget_households`: tập family_id có >=1 cá nhân bị quên —
    dùng để loại đúng các hộ "nhiễm" ra khỏi ứng viên KNN-anchor (xem
    find_knn_retain_groups ở Cell L) — group_prototypes vẫn tính theo
    household, nên loại theo forget_ids (person) sẽ KHÔNG khớp key nào và
    vô tình để lọt household chứa người bị quên vào danh sách anchor được
    bảo vệ (prototype của hộ đó bị nhiễm bởi chính ảnh sắp bị quên)."""
    household_to_indices: Dict[str, List[int]] = {}
    for idx in filtered_indices:
        hh = dataset.households[idx]
        household_to_indices.setdefault(hh, []).append(idx)

    all_households = sorted(household_to_indices.keys())
    random.seed(cfg.seed)
    shuffled_households = all_households.copy()
    random.shuffle(shuffled_households)
    n_unseen = max(1, int(len(shuffled_households) * cfg.unseen_ratio))
    unseen_households = set(shuffled_households[:n_unseen])
    trainable_households = [hh for hh in all_households if hh not in unseen_households]

    unseen_idx = []
    for hh in unseen_households:
        unseen_idx.extend(household_to_indices[hh])

    train_idx, val_idx, test_idx = [], [], []
    for hh in trainable_households:
        idxs = household_to_indices[hh]
        random.shuffle(idxs)
        n = len(idxs)
        n_tr = max(1, int(n * cfg.train_ratio))
        n_va = max(1, int(n * cfg.val_ratio))
        n_te = max(1, n - n_tr - n_va)
        if n_tr + n_va + n_te > n:
            n_tr = max(1, n_tr - 1)
        train_idx.extend(idxs[:n_tr])
        val_idx.extend(idxs[n_tr:n_tr+n_va])
        test_idx.extend(idxs[n_tr+n_va:n_tr+n_va+n_te])

    # ── Chọn CÁ NHÂN bị quên (thay vì cả hộ) ─────────────────────────────
    # v_person-fix: KHÔNG dùng person_id một mình làm khoá — một số bản CSV
    # MUFAC ghi person_id kiểu "số thứ tự trong hộ" (vd 1=con, 2=mẹ, 3=cha),
    # giá trị này LẶP LẠI giữa các hộ khác nhau → nếu chỉ xét person_id thô,
    # số "cá nhân" duy nhất tìm được có thể RẤT ÍT (chỉ vài giá trị), khiến
    # random.sample(..., 20) vô tình chọn HẾT mọi người → retain_train_idx
    # rỗng → DataLoader lỗi "num_samples=0". Ghép family_id + person_id
    # thành khoá "{family_id}::{person_id}" để đảm bảo định danh CÁ NHÂN
    # thật sự duy nhất trên toàn dataset (family+role luôn phân biệt đúng 1
    # người thật, kể cả khi person_id chỉ là số thứ tự nội bộ trong hộ).
    def _person_key(idx: int) -> str:
        return f"{dataset.households[idx]}::{dataset.person_ids[idx]}"

    # Chỉ xét cá nhân có ảnh trong TRAIN (bắt buộc để forget_idx khác rỗng).
    person_to_train_indices: Dict = {}
    for idx in train_idx:
        person_to_train_indices.setdefault(_person_key(idx), []).append(idx)
    eligible_persons = sorted(person_to_train_indices.keys())

    if len(eligible_persons) <= cfg.num_forget_ids:
        raise ValueError(
            f"[split_mufac_by_person] Chỉ tìm được {len(eligible_persons)} cá nhân "
            f"duy nhất trong train, không đủ để chừa lại retain sau khi quên "
            f"{cfg.num_forget_ids} người. Kiểm tra lại cột person_id trong CSV "
            f"(có thể không phải mã định danh cá nhân) hoặc giảm cfg.num_forget_ids."
        )

    random.seed(cfg.seed)
    n_forget_persons = min(cfg.num_forget_ids, len(eligible_persons))
    forget_persons = set(random.sample(eligible_persons, n_forget_persons))

    forget_idx, retain_train_idx = [], []
    for idx in train_idx:
        (forget_idx if _person_key(idx) in forget_persons else retain_train_idx).append(idx)

    retain_test_idx, forget_test_idx = [], []
    for idx in test_idx:
        (forget_test_idx if _person_key(idx) in forget_persons else retain_test_idx).append(idx)

    retain_val_idx, forget_val_idx = [], []
    for idx in val_idx:
        (forget_val_idx if _person_key(idx) in forget_persons else retain_val_idx).append(idx)

    # Hộ gia đình "nhiễm" — chứa >=1 cá nhân bị quên (xem docstring ở trên).
    forget_households = set()
    for idx in forget_idx + forget_test_idx + forget_val_idx:
        forget_households.add(dataset.households[idx])

    return dict(
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        forget_idx=forget_idx, retain_train_idx=retain_train_idx,
        retain_test_idx=retain_test_idx, forget_test_idx=forget_test_idx,
        retain_val_idx=retain_val_idx, forget_val_idx=forget_val_idx,
        forget_ids=sorted(forget_persons),
        forget_households=sorted(forget_households),
        eligible_person_count=len(eligible_persons),   # v_person-fix: để in ra kiểm chứng
        unseen_idx=unseen_idx,
        unseen_households=sorted(unseen_households),
    )

# ===========================================================================
# ▓  CELL D: MODEL ARCHITECTURE — Simple Neck
# ===========================================================================

class ArcMarginProduct(nn.Module):
    """ArcFace margin loss head. Returns cosine-scaled logits."""
    def __init__(self, in_features, out_features, s=32.0, m=0.30):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, emb, labels=None):
        emb = F.normalize(emb, dim=1)
        W   = F.normalize(self.weight, dim=1)
        cos = F.linear(emb, W)            # [B, C]
        if labels is None:
            return cos * self.s
        sin  = torch.sqrt((1.0 - cos**2).clamp(1e-7))
        phi  = cos * self.cos_m - sin * self.sin_m
        phi  = torch.where(cos > self.th, phi, cos - self.mm)
        oh   = torch.zeros_like(cos)
        oh.scatter_(1, labels.view(-1,1), 1.0)
        return (oh * phi + (1.0 - oh) * cos) * self.s


class FaceModel(nn.Module):
    """ResNet-18 + Simple Linear Neck (no BN) + ArcFace.
    forward(x, labels) → (embedding [B, D], logits [B, C])"""
    def __init__(self, num_classes: int, embedding_dim: int = 128,
                 pretrained: bool = True, s: float = 32.0, m: float = 0.30):
        super().__init__()
        bb = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_feat = bb.fc.in_features    # 512 for ResNet-18
        bb.fc   = nn.Identity()
        self.backbone      = bb
        self.embedding_dim = embedding_dim

        # ★ Simple neck — single Linear, no BN
        self.neck = nn.Linear(in_feat, embedding_dim, bias=False)

        self.arcface = ArcMarginProduct(embedding_dim, num_classes, s=s, m=m)

    def forward(self, x, labels=None):
        feat   = self.backbone(x)                # [B, 512]
        emb    = self.neck(feat)                  # [B, D]
        emb    = F.normalize(emb, dim=1)         # unit sphere
        logits = self.arcface(emb, labels)
        return emb, logits


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)

# ===========================================================================
# ▓  CELL E: ORIGINAL MODEL TRAINING
# ===========================================================================

def train_original_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> nn.Module:
    """Warmup tuyến tính, sau đó giảm LR thủ công dựa trên CÙNG MỘT best_val
    dùng cho early-stopping (không dùng ReduceLROnPlateau — scheduler đó
    theo dõi "best" nội bộ riêng, có thể lệch pha với best_val thật nếu đỉnh
    đạt được trong lúc warmup — xem chi tiết trong lpeu_v7.py)."""
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.original_lr,
        momentum=cfg.original_momentum, weight_decay=cfg.original_wd,
        nesterov=True,
    )
    min_lr = cfg.original_lr * 1e-3
    ce = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    epochs_no_improve = 0

    for epoch in range(cfg.original_epochs):
        if epoch < cfg.warmup_epochs:
            warmup_lr = cfg.original_lr * (epoch + 1) / cfg.warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr

        model.train(); total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            _, logits = model(x, y)
            loss = ce(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss)

        val_acc = evaluate_accuracy(model, val_loader, device)

        if val_acc > best_val:
            best_val = val_acc; best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if (epoch >= cfg.warmup_epochs
                    and epochs_no_improve % cfg.lr_reduce_patience == 0):
                for g in optimizer.param_groups:
                    new_lr = max(g['lr'] * 0.5, min_lr)
                    g['lr'] = new_lr
                print(f"[Train]   ↓ LR giảm còn {new_lr:.2e} "
                      f"(no_improve={epochs_no_improve}, best_val={best_val:.4f})")

        cur_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == cfg.original_epochs - 1:
            print(f"[Train][{epoch+1:02d}/{cfg.original_epochs}]  "
                  f"loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}  "
                  f"best_val={best_val:.4f}  no_improve={epochs_no_improve}  lr={cur_lr:.2e}")
        if epochs_no_improve >= cfg.early_stop_patience:
            print(f"[Train] Early stop tại epoch {epoch+1}: "
                  f"val_acc không cải thiện sau {cfg.early_stop_patience} epoch "
                  f"(best_val={best_val:.4f}, lr={cur_lr:.2e})")
            break
    model.load_state_dict(best_state)
    return model


# ===========================================================================
# ▓  CELL F: PROTOTYPE & K-NN UTILITIES (identity-based — dùng cho MUFAC)
# ===========================================================================

@torch.no_grad()
def compute_class_prototypes(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> torch.Tensor:
    """Mean unit-norm embedding per CLASSIFIER class (age-group ở MUFAC).
    Dùng cho retain-prototype anchor (bảo vệ ranh giới phân loại tuổi) —
    KHÔNG dùng cho K-NN 'local identity' (xem compute_group_prototypes)."""
    model.eval()
    D = model.embedding_dim
    sums   = torch.zeros(num_classes, D)
    counts = torch.zeros(num_classes)
    for x, y, _ in loader:
        z, _ = model(x.to(device), None)
        z = z.cpu()
        for i, cls in enumerate(y.tolist()):
            sums[cls] += z[i]
            counts[cls] += 1
    nonzero = counts > 0
    sums[nonzero] = F.normalize(sums[nonzero], dim=1)
    return sums


@torch.no_grad()
def compute_forget_prototype(
    model: nn.Module,
    forget_loader: DataLoader,
    device: str,
) -> torch.Tensor:
    """Unit-norm mean embedding of forget-set samples. Returns CPU tensor."""
    model.eval()
    embs = [model(x.to(device), None)[0].cpu() for x, _, _ in forget_loader]
    z = torch.cat(embs)
    return F.normalize(z.mean(0, keepdim=True), dim=1).squeeze(0)


def identity_equals_class(forget_ids: List, num_classes: int) -> bool:
    """True nếu identity (forget_ids) trùng với classifier class (VGGFace2 /
    Korean Family). False ở MUFAC (forget_ids là family-id dạng str, tách
    biệt khỏi age-class). Dùng để quyết định có áp dụng cơ chế 'stash/freeze
    ArcFace row của lớp bị forget' hay không — cơ chế này chỉ có ý nghĩa khi
    1 lớp ArcFace tương ứng đúng 1 identity."""
    return len(forget_ids) > 0 and all(
        isinstance(f, int) and 0 <= f < num_classes for f in forget_ids
    )


def get_identity_group_ids(dataset) -> List:
    """group_id của từng sample theo index gốc, dùng cho K-NN 'local
    identity'. MUFAC: identity == family_id (dataset.households)."""
    if hasattr(dataset, "households"):
        return dataset.households
    return [y for _, y in dataset.samples]


def compute_group_prototypes(
    model: nn.Module,
    loader: DataLoader,          # phải yield (x, y, orig_idx) — từ IndexedSubset
    group_ids: List,             # group_ids[orig_idx] = group label (family-id)
    device: str,
) -> Dict:
    """Mean unit-norm embedding theo GROUP (identity thật = family)."""
    model.eval()
    sums: Dict = {}
    counts: Dict = {}
    with torch.no_grad():
        for x, _, idx in loader:
            z, _ = model(x.to(device), None)
            z = z.cpu()
            idx_list = idx.tolist() if torch.is_tensor(idx) else list(idx)
            for i, oidx in enumerate(idx_list):
                g = group_ids[oidx]
                if g not in sums:
                    sums[g] = torch.zeros(model.embedding_dim)
                    counts[g] = 0
                sums[g] += z[i]
                counts[g] += 1
    return {g: F.normalize(sums[g].unsqueeze(0), dim=1).squeeze(0)
            for g in sums if counts[g] > 0}


def find_knn_retain_groups(
    forget_proto: torch.Tensor,
    group_prototypes: Dict,
    forget_group_ids: List,
    k: int,
) -> List:
    """K retain identities (family) gần forget_proto nhất."""
    fp = F.normalize(forget_proto.unsqueeze(0), dim=1)
    fset = set(forget_group_ids)
    items = [(g, p) for g, p in group_prototypes.items() if g not in fset]
    if not items:
        return []
    ids = [g for g, _ in items]
    mat = torch.stack([p for _, p in items])
    sims = (mat @ fp.T).squeeze(1)
    k = min(k, len(ids))
    topk = torch.topk(sims, k=k).indices.tolist()
    return [ids[i] for i in topk]


def build_local_anchor_loader_grouped(
    base_dataset,
    retain_train_idx: List[int],
    group_ids: List,
    anchor_group_ids: List,
    batch_size: int,
    num_workers: int,
) -> Optional[DataLoader]:
    """DataLoader chỉ chứa retain-sample thuộc K family gần forget nhất."""
    anchor_set = set(anchor_group_ids)
    anchor_indices = [idx for idx in retain_train_idx if group_ids[idx] in anchor_set]
    if not anchor_indices:
        print("[MUFAC] ⚠ No anchor samples found.")
        return None
    ds = IndexedSubset(base_dataset, anchor_indices)
    print(f"[MUFAC] Anchor loader: {len(anchor_group_ids)} identities, {len(anchor_indices)} samples")
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


# ===========================================================================
# ▓  CELL F2 (MỚI): PROTOTYPE & K-NN THEO TỪNG NGƯỜI
#    Đặt NGAY SAU Cell F.
#
#    Vì sao: bản cũ lấy TRUNG BÌNH embedding của cả 100 người làm 1 prototype
#    duy nhất, rồi tìm 30 HỘ GIA ĐÌNH gần cái trung bình đó. 100 người rải
#    khắp không gian ⇒ trung bình ≈ tâm của cả không gian ⇒ erase không còn
#    "local" (neighbor shift 0.24–0.30, gấp ~60–100× baseline).
#    Bản mới: mỗi ảnh forget bị đẩy khỏi prototype CỦA CHÍNH NGƯỜI ĐÓ, và
#    hàng xóm được bảo vệ là những NGƯỜI retain gần từng người bị quên.
# ===========================================================================

def get_person_keys(dataset) -> List[str]:
    """person_keys[idx] = "{family_id}::{person_id}" — cùng khoá với Cell C."""
    return [f"{hh}::{pid}" for hh, pid in zip(dataset.households, dataset.person_ids)]


def build_person_lookup(person_keys: List[str], forget_persons: List[str]):
    """Trả về (lookup, order):
    lookup[orig_idx] = hàng trong ma trận prototype của người bị quên (-1 nếu không phải forget)
    order            = danh sách forget person theo đúng thứ tự hàng."""
    order = sorted(forget_persons)
    row = {p: i for i, p in enumerate(order)}
    lookup = torch.full((len(person_keys),), -1, dtype=torch.long)
    for i, k in enumerate(person_keys):
        if k in row:
            lookup[i] = row[k]
    return lookup, order


def stack_person_protos(group_protos: Dict, order: List[str]) -> torch.Tensor:
    """[P, D] prototype (unit-norm) theo đúng thứ tự `order`."""
    return torch.stack([F.normalize(group_protos[p], dim=0) for p in order])


@torch.no_grad()
def compute_person_proto_sim(model, loader, lookup, old_protos, device) -> float:
    """Trung bình theo NGƯỜI của cos(prototype hiện tại, prototype gốc).
    Dùng thay cho proto_sim gộp trong vòng lặp LPEU (Migrate→Shatter + chọn checkpoint)."""
    model.eval()
    P, D = old_protos.shape
    sums = torch.zeros(P, D); cnt = torch.zeros(P)
    for x, _, idx in loader:
        z, _ = model(x.to(device), None)
        rows = lookup[idx]
        ok = rows >= 0
        if ok.any():
            sums.index_add_(0, rows[ok], z.cpu()[ok])
            cnt.index_add_(0, rows[ok], torch.ones(int(ok.sum())))
    have = cnt > 0
    cur = F.normalize(sums[have], dim=1)
    return float((cur * F.normalize(old_protos[have], dim=1)).sum(1).mean())


def find_knn_retain_persons(person_protos: Dict, forget_persons: List[str],
                            k_per: int, k_max: int, k_proj: int):
    """Với MỖI người bị quên, lấy k_per người retain gần nhất (cosine).
    Gộp lại, xếp theo độ tương đồng lớn nhất, cắt còn k_max.
    Trả về (anchor_ids, proj_ids): proj_ids = k_proj người gần nhất, dùng cho hard projection."""
    fset = set(forget_persons)
    r_ids = [p for p in person_protos if p not in fset]
    R = F.normalize(torch.stack([person_protos[p] for p in r_ids]), dim=1)
    Fm = F.normalize(torch.stack([person_protos[p] for p in forget_persons if p in person_protos]), dim=1)
    sims = Fm @ R.T                                      # [P_forget, P_retain]
    best = {}
    k = min(k_per, len(r_ids))
    vals, idxs = torch.topk(sims, k=k, dim=1)
    for v_row, i_row in zip(vals.tolist(), idxs.tolist()):
        for v, i in zip(v_row, i_row):
            best[r_ids[i]] = max(best.get(r_ids[i], -2.0), v)
    ranked = sorted(best, key=lambda p: -best[p])
    anchor_ids = ranked[:k_max]
    proj_ids = ranked[:k_proj]
    print(f"  [F2] Neighbors: {len(best)} người retain khác nhau → giữ {len(anchor_ids)} làm anchor, "
          f"{len(proj_ids)} cho projection | sim max={best[ranked[0]]:.3f} min(anchor)={best[anchor_ids[-1]]:.3f}")
    return anchor_ids, proj_ids


@torch.no_grad()
def per_sample_losses(model, loader, device, group_keys: List[str]):
    """Loss CE của TỪNG ảnh + khoá nhóm (người/hộ) — lưu lại để làm
    identity-level paired bootstrap sau này (R2 yêu cầu)."""
    model.eval(); ce = nn.CrossEntropyLoss(reduction='none')
    L, G = [], []
    for x, y, idx in loader:
        _, logits = model(x.to(device), None)
        L.append(ce(logits, y.to(device)).cpu())
        G.extend(group_keys[i] for i in idx.tolist())
    return torch.cat(L).numpy(), np.array(G)


# ===========================================================================
# ▓  CELL G: EWC UTILITIES
# ===========================================================================

@torch.no_grad()
def snapshot_neck_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Save a copy of neck (and layer4) parameter values as θ* for EWC."""
    snap = {}
    for name, param in model.named_parameters():
        if "neck" in name or "backbone.layer4" in name:
            snap[name] = param.data.clone().cpu()
    return snap


def compute_ewc_fisher(
    model: nn.Module,
    retain_loader: DataLoader,
    device: str,
    n_batches: int = 30,
) -> Dict[str, torch.Tensor]:
    """
    Diagonal Fisher Information trên retain set cho neck + layer4 params.
    F_i = (1/N) Σ (∂CE/∂θ_i)² — F_i cao → θ_i quan trọng cho retain → EWC
    phạt nặng nếu θ_i lệch khỏi θ*.
    """
    model.eval()
    fisher: Dict[str, torch.Tensor] = {}
    target_names = set()
    for name, param in model.named_parameters():
        if "neck" in name or "backbone.layer4" in name:
            # v7.4-fix: fisher phải khởi tạo trên CPU vì vòng lặp bên dưới
            # cộng dồn param.grad.data.pow(2).cpu() — nếu model đang ở cuda,
            # torch.zeros_like(param.data) tạo tensor cuda, còn RHS là cpu
            # → "Expected all tensors to be on the same device" khi +=.
            fisher[name] = torch.zeros_like(param.data, device='cpu')
            target_names.add(name)

    ce = nn.CrossEntropyLoss()
    count = 0
    for x, y, _ in retain_loader:
        if count >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        _, logits = model(x, y)
        loss = ce(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if name in target_names and param.grad is not None:
                fisher[name] += param.grad.data.pow(2).cpu()
        count += 1

    for name in fisher:
        fisher[name] /= max(count, 1)

    total_params = sum(v.numel() for v in fisher.values())
    mean_f = sum(float(v.mean()) for v in fisher.values()) / max(len(fisher), 1)
    print(f"[MUFAC] EWC Fisher computed: {len(fisher)} param groups, "
          f"{total_params:,} params, mean_F={mean_f:.6f}")
    return fisher


def compute_ewc_loss(
    model: nn.Module,
    theta_star: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
    w_ewc: float,
    device: str,
) -> torch.Tensor:
    """L_ewc = (w_ewc/2) × Σ_i F_i × (θ_i − θ*_i)²"""
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for name, param in model.named_parameters():
        if name in fisher and name in theta_star:
            F_i      = fisher[name].to(device)
            theta_i  = theta_star[name].to(device)
            loss = loss + (F_i * (param - theta_i).pow(2)).sum()
    return (w_ewc / 2.0) * loss

# ===========================================================================
# ▓  CELL H: LOSS FUNCTIONS
# ===========================================================================

class ForgetLossV7(nn.Module):
    """Three-component forget loss. repel_margin = -0.30: pair repulsion
    đẩy tới khi cos(z_fi, z_fj) < -0.3 — scatter thật, không chỉ decorrelate."""
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.cfg   = cfg
        self.C     = num_classes
        self.log_C = math.log(num_classes)

    def forward(
        self,
        z_f:       torch.Tensor,   # [B, D] unit-norm forget embeddings
        old_proto: torch.Tensor,   # [D]    frozen prototype (on device)
        logits_f:  torch.Tensor,   # [B, C] ArcFace logits
        w_erase:   float = None,   # override for staged schedule
        w_repel:   float = None,
        w_kl_uni:  float = None,   # v_B — override for staged Migrate→Shatter schedule
        w_div:     float = None,   # v_C — override for staged Migrate→Shatter schedule
        logits_f_unseen_old: torch.Tensor = None,  # v_C — [B,C] logits của old_model trên batch unseen cùng age-class
    ) -> Tuple[torch.Tensor, Dict]:
        B = z_f.size(0)
        w_e  = w_erase  if w_erase  is not None else self.cfg.w_erase
        w_r  = w_repel  if w_repel  is not None else self.cfg.w_repel
        w_kl = w_kl_uni if w_kl_uni is not None else self.cfg.w_kl_uniform
        w_d  = w_div    if w_div    is not None else self.cfg.w_div

        # ── Anti-prototype push ────────────────────────────────────────────
        # resubmit: old_proto có thể là [D] (gộp, bản cũ) hoặc [B, D] (prototype riêng từng người)
        _p = old_proto if old_proto.dim() == 2 else old_proto.unsqueeze(0)
        L_erase = (z_f * _p).sum(dim=1).mean()

        # ── KL-to-Uniform (calibration) ───────────────────────────────────
        p        = F.softmax(logits_f, dim=1)
        H        = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
        L_kl_uni = self.log_C - H

        # ── Pair repulsion with aggressive margin ─────────────────────────
        if B > 1:
            sim_mat = z_f @ z_f.T                  # [B, B]
            mask    = ~torch.eye(B, dtype=torch.bool, device=z_f.device)
            L_repel = F.relu(sim_mat[mask] - self.cfg.repel_margin).mean()
        else:
            L_repel = torch.zeros(1, device=z_f.device).squeeze()

        # ── Hướng C: KL-to-unseen (class-matched distillation) ────────────
        # v_C — KL(P_orig(x_unseen) ‖ P_forget(x_forget)), cùng cách dùng
        # F.kl_div như RetainLossV7.forward (Cell H, KD term ở dưới): target
        # = p_orig_unseen (no-grad, từ old_model), input =
        # log_softmax(logits_f) (CÓ grad) — minimize kéo p_forget →
        # p_orig_unseen. logits_f và logits_f_unseen_old không cùng ảnh, chỉ
        # cùng age-class (matched theo nhãn thật của batch forget — xem
        # _sample_matched_unseen_batch, Cell I) nên so hợp lệ mà KHÔNG rò
        # rỉ ảnh unseen vào forget batch.
        if logits_f_unseen_old is not None and w_d > 0:
            log_p_f       = F.log_softmax(logits_f, dim=1)
            p_orig_unseen = F.softmax(logits_f_unseen_old, dim=1)
            L_div = F.kl_div(log_p_f, p_orig_unseen, reduction='batchmean')
        else:
            L_div = torch.zeros(1, device=z_f.device).squeeze()

        total = (w_e * L_erase
               + w_kl * L_kl_uni
               + w_r  * L_repel
               + w_d  * L_div)

        parts = {
            "L_erase":      float(L_erase.detach()),
            "L_kl_uni":     float(L_kl_uni.detach()),
            "L_repel":      float(L_repel.detach()),
            "L_div":        float(L_div.detach()),
            "entropy":      float(H.detach()),
            "cos_to_proto": float(L_erase.detach()),
        }
        return total, parts


class RetainLossV7(nn.Module):
    """Retain loss (KD + K-NN local anchor + CE) + retain-prototype anchor.

    L_proto_retain là đối xứng trực tiếp với L_erase của ForgetLossV7:
      L_erase        = mean  cos(z_f, p_forget_old)   → ĐẨY forget ra xa prototype
      L_proto_retain = mean(1 - cos(z_r, p_{y_r}))    → KÉO retain VỀ prototype của
                                                          đúng lớp của nó
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.T   = cfg.kd_temperature
        self.ce  = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_r:        torch.Tensor,
        logits_r_old:    torch.Tensor,
        y_r:             torch.Tensor,
        z_n_new:         Optional[torch.Tensor] = None,
        z_n_old:         Optional[torch.Tensor] = None,
        w_local:         float = None,   # override for staged schedule
        z_r_new:         Optional[torch.Tensor] = None,   # [B, D] retain embeddings
        retain_prototypes: Optional[torch.Tensor] = None, # [C, D] frozen, trên device
        w_proto:         float = None,   # override for staged schedule
    ) -> Tuple[torch.Tensor, Dict]:
        w_l = w_local if w_local is not None else self.cfg.w_kd_local
        w_p = w_proto if w_proto is not None else self.cfg.w_proto_retain

        # Global KL distillation
        log_p = F.log_softmax(logits_r     / self.T, dim=1)
        p_old = F.softmax(   logits_r_old  / self.T, dim=1)
        kl    = (self.T ** 2) * F.kl_div(log_p, p_old, reduction='batchmean')

        # CE
        ce = self.ce(logits_r, y_r)

        # Local K-NN cosine anchor (gated by w_l — zero in Phase 1)
        if z_n_new is not None and z_n_old is not None and w_l > 0:
            cos_sim  = (z_n_new * z_n_old).sum(dim=1)
            L_anchor = (1.0 - cos_sim).mean()
        else:
            L_anchor = torch.zeros(1, device=logits_r.device).squeeze()

        # Retain-prototype anchor (luôn bật, kể cả Phase 1)
        if (self.cfg.use_proto_retain and z_r_new is not None
                and retain_prototypes is not None and w_p > 0):
            proto_y  = retain_prototypes[y_r]            # [B, D]
            cos_p    = (z_r_new * proto_y).sum(dim=1)
            L_proto  = (1.0 - cos_p).mean()
        else:
            L_proto = torch.zeros(1, device=logits_r.device).squeeze()

        total = (self.cfg.w_kd_global * kl
               + w_l                  * L_anchor
               + self.cfg.w_ce_retain * ce
               + w_p                  * L_proto)

        parts = {
            "L_kd":     float(kl.detach()),
            "L_anchor": float(L_anchor.detach()) if isinstance(L_anchor, torch.Tensor) else 0.0,
            "L_ce":     float(ce.detach()),
            "L_proto":  float(L_proto.detach()) if isinstance(L_proto, torch.Tensor) else 0.0,
        }
        return total, parts

# ===========================================================================
# ▓  CELL H2: PCGrad UTILITY
# ===========================================================================
 
def pcgrad_project(g_f: torch.Tensor, g_r: torch.Tensor) -> torch.Tensor:
    """
    Project g_f onto orthogonal complement of g_r when conflicting.
    g_f' = g_f − (⟨g_f,g_r⟩/‖g_r‖²)·g_r  if ⟨g_f,g_r⟩ < 0, else g_f.
    """
    inner = torch.dot(g_f, g_r)
    if inner >= 0:
        return g_f
    g_r_sq = torch.dot(g_r, g_r)
    if g_r_sq < 1e-12:
        return g_f
    return g_f - (inner / g_r_sq) * g_r
 
 
def build_orthonormal_basis(proto_matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """v7.5 — proto_matrix: [K, D] (K local retain-prototype véc-tơ, không
    nhất thiết trực giao với nhau). Trả về basis trực giao hoá [K', D] (hàng
    trực giao đơn vị, K' <= K) trương ra CÙNG không gian con — dùng QR trên
    ma trận chuyển vị. Trả None nếu proto_matrix rỗng/None (tắt gradient
    projection một cách an toàn — fallback về hành vi cũ)."""
    if proto_matrix is None or proto_matrix.numel() == 0:
        return None
    Q, _ = torch.linalg.qr(proto_matrix.T, mode='reduced')   # Q: [D, K']
    return Q.T.contiguous()                                   # [K', D]
 
 
def project_out_prototype_directions(
    grad:        torch.Tensor,   # [D, H] — gradient của neck.weight
    proto_basis: torch.Tensor,   # [K', D] trực giao đơn vị, cùng device với grad
) -> torch.Tensor:
    """v7.5 — GRADIENT PROJECTION (Local Prototype Erasure theo đúng nghĩa
    đen của tên phương pháp).
 
    neck.weight ánh xạ trực tiếp feat → embedding: emb = W @ feat. Với BẤT KỲ
    feat nào, thay đổi embedding do một update ΔW gây ra là ΔW @ feat ∈ R^D.
    Ta muốn thành phần của thay đổi đó dọc theo mỗi local retain-prototype
    p_k LUÔN BẰNG 0 — tức p_k^T @ ΔW = 0 (một vector H-chiều) với mọi k. Đây
    là một ràng buộc CỨNG ở cấp tham số (không phải cấp loss), nên không bị
    'đánh đổi ngược' qua các bước train sau như cách các loss-term vẫn bị
    (khác PCGrad — vốn chỉ giải quyết xung đột theo batch retain HIỆN TẠI,
    nhiễu theo batch và không cố định qua thời gian).
 
    Với MUFAC: local retain-prototypes là các gia đình (household) K-NN gần
    forget-household nhất trong không gian embedding — không liên quan tới
    age-class. Ràng buộc này bảo vệ đúng cấu trúc 'local identity' mà LPEU
    hướng tới, tách biệt khỏi classifier task (age).
 
    Cài đặt: coi mỗi CỘT của grad (∈ R^D, D=embedding_dim) là một vector cần
    chiếu lên phần bù trực giao của span(proto_basis):
        grad_perp = grad − proto_basis^T @ (proto_basis @ grad)
    """
    proj = proto_basis.T @ (proto_basis @ grad)   # [D, H]
    return grad - proj

# ===========================================================================
# ▓  CELL I: STAGED UNLEARNING LOOP — LPEU   [resubmit: thêm chế độ per-person]
# ===========================================================================
 
def get_stage_weights(epoch: int, cfg: Config) -> Tuple[float, float, float, float]:
    """
    Phase 1 — ERASE (epoch < stage1_epochs): Anchor OFF. Max forgetting.
    Phase 2 — REFINE (stage1 ≤ epoch < stage1+stage2): Anchor 50%.
    Phase 3 — STABLE (epoch ≥ stage1+stage2): Anchor 100%. Light forgetting.

    v7.8 — GHI CHÚ: hàm này giờ chỉ còn quyết định w_local (anchor) và boost.
    w_erase/w_repel trả về ở đây KHÔNG còn được dùng trực tiếp trong vòng lặp
    forget nữa (giữ lại trong return để không phá vỡ signature/chỗ gọi khác)
    — 2 giá trị đó nay do get_forget_subphase_weights() quyết định, theo
    lịch Migrate→Shatter tách riêng khỏi lịch anchor.
    """
    e1 = cfg.stage1_epochs
    e2 = cfg.stage1_epochs + cfg.stage2_epochs
 
    if epoch < e1:
        return 0.0, cfg.w_erase, cfg.w_repel, cfg.pcgrad_boost
    elif epoch < e2:
        return cfg.w_kd_local * 0.5, cfg.w_erase * 0.75, cfg.w_repel * 0.67, cfg.pcgrad_boost
    else:
        return cfg.w_kd_local, cfg.w_erase * 0.5, cfg.w_repel * 0.33, cfg.pcgrad_boost * 0.5


def get_forget_subphase_weights(
    epoch: int,
    proto_drop_so_far: float,
    initial_proto_sim: float,
    in_shatter: bool,
    cfg: Config,
) -> Tuple[float, float, float, float, bool]:
    """
    v7.8 — Migrate → Shatter: tách lịch w_erase / w_repel khỏi lịch anchor
    (get_stage_weights ở trên). 2 sub-phase NỐI TIẾP, 1 chiều:

      MIGRATE  (in_shatter=False): w_erase=cfg.w_erase đầy đủ, w_repel=0.
               Dồn hết lực "di chuyển" cụm forget ra khỏi prototype cũ,
               không bị L_repel giành lực / lấp lại vùng vừa dọn.
      SHATTER  (in_shatter=True):  w_erase hạ còn một sàn dư
               (shatter_erase_floor × cfg.w_erase, chống trôi ngược),
               w_repel=cfg.w_repel đầy đủ — giờ mới "phá rã" cụm đã dịch.

    Điều kiện chuyển MIGRATE → SHATTER (đánh giá 1 lần mỗi epoch, dùng
    proto_drop đo được ở CUỐI epoch trước — proto_drop_so_far):
      - proto_drop_so_far / (1 + initial_proto_sim) ≥ migrate_target_fraction
        (đã đạt đủ % mức giảm lý thuyết tối đa), HOẶC
      - epoch ≥ migrate_max_epochs (an toàn, tránh kẹt MIGRATE vô hạn nếu
        proto_drop không đạt ngưỡng vì lý do khác — vd anchor/EWC cản trở).
    Một khi đã sang SHATTER thì KHÔNG quay lại MIGRATE (in_shatter chỉ có
    thể False→True, không có chiều ngược).

    v_B — THÊM w_kl_uni vào cùng lịch Migrate→Shatter (trước đây L_kl_uni
    chạy full-weight suốt, không được stage hoá — nay: 0 trong MIGRATE,
    cfg.w_kl_uniform đủ trong SHATTER). Lý do: L_kl_uni là cơ chế "ép giảm
    task accuracy trên ảnh forget" (Hướng B) — để nó chạy song song ngay từ
    đầu MIGRATE sẽ cạnh tranh lực với L_erase (vốn đang cố di chuyển cụm ra
    khỏi prototype cũ thuần trong embedding-space); dồn nó vào SHATTER, cùng
    lúc với L_repel, cho nhất quán về mặt giai đoạn.
    """
    max_possible_drop = 1.0 + initial_proto_sim
    achieved_fraction = (
        proto_drop_so_far / max_possible_drop if max_possible_drop > 1e-8 else 1.0
    )

    if not in_shatter:
        crossed_threshold = achieved_fraction >= cfg.migrate_target_fraction
        crossed_epoch_cap = epoch >= cfg.migrate_max_epochs
        if crossed_threshold or crossed_epoch_cap:
            in_shatter = True

    if not in_shatter:
        w_erase  = cfg.w_erase
        w_repel  = 0.0
        w_kl_uni = 0.0   # v_B — Hướng B: chưa ép task-accuracy trong MIGRATE
        w_div    = 0.0   # v_C — chưa ép KL-to-unseen trong MIGRATE, cùng lý do
    else:
        w_erase  = cfg.w_erase * cfg.shatter_erase_floor
        w_repel  = cfg.w_repel
        w_kl_uni = cfg.w_kl_uniform   # v_B — bật đủ lực trong SHATTER
        w_div    = cfg.w_div          # v_C — bật đủ lực trong SHATTER

    return w_erase, w_repel, w_kl_uni, w_div, in_shatter


def _sample_matched_unseen_batch(
    y_f: torch.Tensor,
    unseen_by_class_images: Dict[int, torch.Tensor],
    device,
) -> torch.Tensor:
    """v_C — Với mỗi nhãn (age-class) trong y_f, lấy NGẪU NHIÊN 1 ảnh unseen
    CÙNG age-class (có hoàn lại — một số lớp unseen chỉ có vài chục ảnh,
    batch_size forget có thể lớn hơn số ảnh unseen của lớp đó). Trả về
    tensor [B, C, H, W] CÙNG THỨ TỰ với y_f, để ghép với logits_f theo từng
    vị trí trong batch. Không rò rỉ ảnh forget: nguồn là unseen_idx, tách
    biệt hoàn toàn khỏi train/forget (xem split_mufac_by_person, Cell C)."""
    imgs = []
    for y in y_f.tolist():
        pool = unseen_by_class_images.get(y)
        if pool is None or pool.size(0) == 0:
            pool = next(iter(unseen_by_class_images.values()))  # fallback hiếm gặp
        j = random.randint(0, pool.size(0) - 1)
        imgs.append(pool[j])
    return torch.stack(imgs).to(device)


def run_lpeu_v7_unlearning(
    model:              nn.Module,
    old_model:          nn.Module,
    forget_loader:      DataLoader,
    retain_loader:      DataLoader,
    retain_test_loader: DataLoader,
    anchor_loader:      Optional[DataLoader],
    forget_ids:         List,
    num_classes:        int,
    cfg:                Config,
    retain_acc_ref:     float,
    fisher:             Optional[Dict[str, torch.Tensor]] = None,
    theta_star:         Optional[Dict[str, torch.Tensor]] = None,
    retain_prototypes:  Optional[torch.Tensor] = None,   # [C, D] frozen, CPU
    local_retain_prototypes: Optional[torch.Tensor] = None,  # [K, D] frozen, CPU — v7.5 gradient projection
    unseen_by_class_images: Optional[Dict[int, torch.Tensor]] = None,  # v_C — {age_class: [n,C,H,W] CPU}, cho L_div
    person_lookup:     Optional[torch.Tensor] = None,  # resubmit — [N_dataset] → hàng prototype người bị quên (-1 nếu không)
    person_protos_old: Optional[torch.Tensor] = None,  # resubmit — [P, D] prototype gốc của TỪNG người bị quên
) -> nn.Module:
    """
    LPEU staged unlearning loop.
 
    Per-iteration:
    ═══════════════════════════════════════════════════════════════
    [A] RETAIN STEP:
      1. Forward retain batch → logits_r
      2. Forward anchor batch → z_n_new (Phase 2+3 only)
      3. L_retain = KL + staged_anchor + CE  [+ EWC if use_ewc]
      4. backward → save neck grads for PCGrad → step → STASH ArcFace
 
    [B] FORGET STEP (PCGrad protected):
      1. Forward forget batch → (z_f, logits_f)
      2. L_forget = staged_erase + kl_uni + staged_repel
      3. backward → PCGrad project neck grads → boost → step
      4. RESTORE ArcFace retain weights
    ═══════════════════════════════════════════════════════════════
    """
    device = cfg.device
    model.to(device)
    old_model.to(device)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad = False
 
    # ── Trainable parameters ──────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False
 
    for name, p in model.named_parameters():
        if "neck" in name or "arcface" in name:
            p.requires_grad = True
 
    layer4_params = []
    for name, p in model.named_parameters():
        if "backbone.layer4" in name:
            p.requires_grad = True
            layer4_params.append(p)
 
    neck_arcface_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and "backbone.layer4" not in n]
    all_trainable = [p for p in model.parameters() if p.requires_grad]
 
    pcgrad_param_names = set(n for n, p in model.named_parameters()
                             if p.requires_grad and ("neck" in n or "backbone.layer4" in n))
 
    n_train = sum(p.numel() for p in all_trainable)
    print(f"[MUFAC] Trainable: {n_train:,} params  |  "
          f"EWC: {'ON' if cfg.use_ewc and fisher else 'OFF'}")
 
    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": neck_arcface_params, "lr": cfg.unlearn_lr},
            {"params": layer4_params,       "lr": cfg.unlearn_lr * cfg.layer4_lr_mult},
        ],
        weight_decay=cfg.unlearn_wd,
        betas=(0.9, 0.999),
    )
    total_epochs = cfg.unlearn_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=cfg.unlearn_lr * 0.05,
    )
 
    forget_crit = ForgetLossV7(cfg, num_classes)
    retain_crit = RetainLossV7(cfg)
 
    forget_proto = compute_forget_prototype(old_model, forget_loader, device)
    forget_proto = forget_proto.to(device)
    _p_cpu       = forget_proto.cpu()
    initial_proto_sim = float(
        compute_forget_prototype(old_model, forget_loader, device).dot(_p_cpu)
    )

    # ── resubmit: chế độ PER-PERSON ──────────────────────────────────────
    # Mỗi ảnh forget bị đẩy khỏi prototype của CHÍNH NGƯỜI ĐÓ (không phải
    # trung bình của cả 100 người). proto_sim/proto_drop cũng đo theo từng
    # người rồi lấy trung bình — dùng cho Migrate→Shatter và chọn checkpoint.
    _per_person = (person_lookup is not None and person_protos_old is not None)
    if _per_person:
        person_lookup_dev = person_lookup.to(device)
        person_protos_dev = F.normalize(person_protos_old, dim=1).to(device)
        initial_proto_sim = compute_person_proto_sim(
            old_model, forget_loader, person_lookup, person_protos_old, device)
        print(f"[resubmit] PER-PERSON erase ON — {person_protos_old.shape[0]} prototypes, "
              f"initial per-person proto_sim={initial_proto_sim:.4f}")

    # v_B — HƯỚNG B: checkpoint selection (bên dưới) trước đây CHỈ nhìn
    # proto_drop + retain_score, không hề biết L_kl_uni đang cố ép fgt_acc
    # giảm về gần rc (1/num_classes) — nên checkpoint cuối cùng được chọn có
    # thể hoàn toàn không phản ánh hiệu ứng L_kl_uni tạo ra trong lúc train.
    # Thêm mốc fgt_acc BAN ĐẦU (của old_model, trước khi unlearn) để đo %
    # tiến gần tới rc mỗi epoch — dùng làm tín hiệu forget bổ sung.
    initial_fgt_acc = evaluate_accuracy(old_model, forget_loader, device)
    rc = 1.0 / num_classes
 
    # Retain-prototype anchor: frozen [C, D] prototypes tính TRƯỚC unlearning
    retain_proto_dev = (
        retain_prototypes.to(device)
        if (cfg.use_proto_retain and retain_prototypes is not None) else None
    )
 
    # v7.5 — Gradient projection setup: basis trực giao từ local KNN retain
    # prototypes (K gia đình retain gần forget-household nhất). Chỉ áp dụng
    # lên neck.weight — lớp linear cuối sinh embedding TRỰC TIẾP — nơi phép
    # chiếu có ý nghĩa hình học rõ ràng (xem project_out_prototype_directions).
    proto_basis_dev = None
    if cfg.use_grad_projection and local_retain_prototypes is not None:
        proto_basis_dev = build_orthonormal_basis(local_retain_prototypes)
        if proto_basis_dev is not None:
            proto_basis_dev = proto_basis_dev.to(device)
            print(f"[v7] Gradient projection ON — basis rank={proto_basis_dev.shape[0]} "
                  f"(từ {local_retain_prototypes.shape[0]} local prototypes)")
 
    # MUFAC: identity (family) tách biệt khỏi classifier class (age). Cơ chế
    # stash/restore ArcFace row chỉ có ý nghĩa khi 1 lớp ArcFace == 1 identity
    # (VGGFace2/Korean Family). Ở MUFAC, coi TOÀN BỘ class là "retain" — tức
    # đóng băng nguyên đầu ArcFace khỏi forget-step (hợp lý: ranh giới phân
    # loại tuổi không nên đổi chỉ vì quên 1 gia đình — phần "xoá
    # representation" thật sự nằm ở neck/layer4, không bị ảnh hưởng).
    if identity_equals_class(forget_ids, num_classes):
        fset = set(forget_ids)
        retain_cls_t = torch.tensor(
            [c for c in range(num_classes) if c not in fset], dtype=torch.long
        )
    else:
        retain_cls_t = torch.arange(num_classes, dtype=torch.long)
 
    best_state   = copy.deepcopy(model.state_dict())
    best_balance = -1.0
    violation_streak = 0
 
    print(f"\n[MUFAC] Starting staged unlearning — {total_epochs} epochs total")
    print(f"     Anchor schedule   — Phase 1 (ERASE):  epoch 0–{cfg.stage1_epochs-1}      anchor=OFF")
    print(f"     Anchor schedule   — Phase 2 (REFINE): epoch {cfg.stage1_epochs}–{cfg.stage1_epochs+cfg.stage2_epochs-1}  anchor=50%")
    print(f"     Anchor schedule   — Phase 3 (STABLE): epoch {cfg.stage1_epochs+cfg.stage2_epochs}–{total_epochs-1}  anchor=100%")
    print(f"     Forget schedule   — MIGRATE (w_erase={cfg.w_erase}, w_repel=0, w_div=0) → "
          f"SHATTER (w_erase={cfg.w_erase*cfg.shatter_erase_floor:.2f}, w_repel={cfg.w_repel}, w_div={cfg.w_div}), "
          f"switch @ proto_drop≥{cfg.migrate_target_fraction:.0%} of max or epoch≥{cfg.migrate_max_epochs}")
    print(f"     v_C: L_div (KL-to-unseen, class-matched) — "
          f"{'unseen_by_class_images đã truyền, sẽ tính L_div trong SHATTER' if unseen_by_class_images is not None else 'KHÔNG truyền unseen_by_class_images — L_div sẽ luôn=0 (Hướng C tắt)'}")
    print(f"     Initial proto_sim = {initial_proto_sim:.4f}")
 
    anchor_iter = iter(anchor_loader) if anchor_loader else None
    in_shatter      = False   # Migrate→Shatter state — 1 chiều, False→True
    proto_drop_prev = 0.0     # proto_drop đo cuối epoch TRƯỚC (epoch 0 dùng 0.0 → luôn MIGRATE)
 
    for epoch in range(total_epochs):
        model.train()
 
        # ── Staged weight schedule ────────────────────────────────────────
        # w_local (anchor) / boost: lịch cũ theo epoch (get_stage_weights).
        # w_erase / w_repel: lịch Migrate→Shatter, TÁCH RIÊNG, theo proto_drop
        # đo được (không theo epoch cố định) — xem get_forget_subphase_weights.
        w_local, _, _, boost = get_stage_weights(epoch, cfg)
        w_erase, w_repel, w_kl_uni, w_div, in_shatter = get_forget_subphase_weights(
            epoch, proto_drop_prev, initial_proto_sim, in_shatter, cfg
        )
        phase_name = (
            "ERASE" if epoch < cfg.stage1_epochs else
            "REFINE" if epoch < cfg.stage1_epochs + cfg.stage2_epochs else
            "STABLE"
        )
        subphase_name = "SHATTER" if in_shatter else "MIGRATE"
 
        S = {k: 0. for k in ["kd","anchor","ce","proto","erase","kl_uni","repel","div","entropy","ewc","n_conf","steps"]}
        forget_iter = iter(forget_loader)
 
        for x_r, y_r, _ in retain_loader:
            x_r, y_r = x_r.to(device), y_r.to(device)
 
            # ══════════════════════════════════════════════════════════════
            # [A] RETAIN STEP
            # ══════════════════════════════════════════════════════════════
            z_r, logits_r = model(x_r, y_r)
            with torch.no_grad():
                _, logits_r_old = old_model(x_r, y_r)
 
            # Anchor batch (only in Phase 2+3 when w_local > 0)
            z_n_new = z_n_old = None
            if anchor_iter is not None and w_local > 0:
                try:
                    x_n, _, _ = next(anchor_iter)
                except StopIteration:
                    anchor_iter = iter(anchor_loader)
                    x_n, _, _ = next(anchor_iter)
                x_n = x_n.to(device)
                z_n_new, _ = model(x_n, None)
                with torch.no_grad():
                    z_n_old, _ = old_model(x_n, None)
 
            loss_r, parts_r = retain_crit(
                logits_r, logits_r_old, y_r, z_n_new, z_n_old,
                w_local=w_local,
                z_r_new=z_r, retain_prototypes=retain_proto_dev,
            )
 
            # Add EWC penalty (acts as data-free retain guard in Phase 1)
            ewc_val = 0.0
            if cfg.use_ewc and fisher is not None and theta_star is not None:
                L_ewc  = compute_ewc_loss(model, theta_star, fisher, cfg.w_ewc, device)
                loss_r = loss_r + L_ewc
                ewc_val = float(L_ewc.detach())
 
            optimizer.zero_grad()
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip_retain)
 
            # Save retain grads for PCGrad
            retain_grads: Dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if name in pcgrad_param_names and param.grad is not None:
                    retain_grads[name] = param.grad.detach().clone()
 
            optimizer.step()
 
            # Stash retain ArcFace weights
            stashed = model.arcface.weight.data[retain_cls_t].clone().cpu()
 
            S["kd"]     += parts_r["L_kd"]
            S["anchor"] += parts_r["L_anchor"]
            S["ce"]     += parts_r["L_ce"]
            S["proto"]  += parts_r["L_proto"]
            S["ewc"]    += ewc_val
 
            # ══════════════════════════════════════════════════════════════
            # [B] FORGET STEP — PCGrad protected
            # ══════════════════════════════════════════════════════════════
            try:
                x_f, y_f, idx_f = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                x_f, y_f, idx_f = next(forget_iter)
            x_f = x_f.to(device)
            # resubmit — prototype của đúng người trong từng ảnh ([B, D]) hoặc prototype gộp cũ ([D])
            proto_arg = (person_protos_dev[person_lookup_dev[idx_f.to(device)]]
                         if _per_person else forget_proto)
 
            # v_C — Hướng C: batch unseen "khớp lớp" (matched theo age-class
            # thật của x_f) để tính L_div bên dưới. Chỉ tính khi w_div>0
            # (SHATTER) — tránh forward thừa trong MIGRATE.
            logits_f_unseen_old = None
            if unseen_by_class_images is not None and w_div > 0:
                x_u_matched = _sample_matched_unseen_batch(y_f, unseen_by_class_images, device)
                with torch.no_grad():
                    _, logits_f_unseen_old = old_model(x_u_matched, None)

            z_f, logits_f = model(x_f, None)
            loss_f, parts_f = forget_crit(
                z_f, proto_arg, logits_f,
                w_erase=w_erase,
                w_repel=w_repel,
                w_kl_uni=w_kl_uni,
                w_div=w_div,
                logits_f_unseen_old=logits_f_unseen_old,
            )
 
            optimizer.zero_grad()
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip_forget)
 
            # ── PCGrad surgery on neck + layer4 ────────────────────────────
            # Bước forget áp dụng SGD thủ công (param -= lr*grad) trực tiếp
            # lên neck/layer4 SAU khi PCGrad chiếu + nhân boost, KHÔNG cho
            # AdamW cập nhật 2 nhóm này (AdamW chuẩn hoá theo m/√v khiến
            # boost gần như vô tác dụng nếu để AdamW tự step — xem lpeu_v7.py).
            # ArcFace vẫn update qua AdamW như cũ.
            n_conflicts = 0
            manual_grads: Dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if name not in pcgrad_param_names or param.grad is None:
                    continue
                g_f_flat = param.grad.detach().flatten()
                if name in retain_grads:
                    g_r_flat = retain_grads[name].flatten()
                    if torch.dot(g_f_flat, g_r_flat) < 0:
                        n_conflicts += 1
                    g_f_proj = pcgrad_project(g_f_flat, g_r_flat)
                else:
                    g_f_proj = g_f_flat
                manual_grads[name] = (g_f_proj * boost).reshape(param.shape)
                param.grad = None   # chặn AdamW cập nhật tham số này qua step()
 
            # v7.5 — Gradient projection: loại bỏ khỏi neck.weight mọi thành
            # phần update nằm trong span(local retain-prototypes) — ràng buộc
            # hình học cứng, độc lập với batch hiện tại (bổ sung cho PCGrad ở
            # trên, vốn chỉ xử lý xung đột theo batch).
            if proto_basis_dev is not None and "neck.weight" in manual_grads:
                manual_grads["neck.weight"] = project_out_prototype_directions(
                    manual_grads["neck.weight"], proto_basis_dev
                )
 
            optimizer.step()   # chỉ còn tác dụng lên ArcFace (và các param khác nếu có)
 
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in manual_grads:
                        lr_mult = cfg.layer4_lr_mult if "backbone.layer4" in name else 1.0
                        param -= cfg.forget_manual_lr * lr_mult * manual_grads[name]
 
            # Restore retain ArcFace weights
            model.arcface.weight.data[retain_cls_t] = stashed.to(device)
 
            S["erase"]   += parts_f["L_erase"]
            S["kl_uni"]  += parts_f["L_kl_uni"]
            S["repel"]   += parts_f["L_repel"]
            S["div"]     += parts_f["L_div"]
            S["entropy"] += parts_f["entropy"]
            S["n_conf"]  += n_conflicts
            S["steps"]   += 1
 
            # ══════════════════════════════════════════════════════════════
            # [C] IN-LOOP RETAIN REPAIR
            # ══════════════════════════════════════════════════════════════
            # Sau mỗi forget step, chạy ngay một mini retain KD step để
            # "vá" damage từ forget gradient trước khi chuyển batch tiếp.
            if cfg.use_inloop_repair:
                _, logits_r_rep = model(x_r, y_r)
                with torch.no_grad():
                    _, logits_r_old_rep = old_model(x_r, y_r)
                log_p_rep = F.log_softmax(logits_r_rep / cfg.kd_temperature, dim=1)
                p_old_rep = F.softmax(logits_r_old_rep / cfg.kd_temperature, dim=1)
                loss_rep  = cfg.w_repair * (cfg.kd_temperature ** 2) * \
                            F.kl_div(log_p_rep, p_old_rep, reduction='batchmean')
                optimizer.zero_grad()
                loss_rep.backward()
                torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip_retain)
                optimizer.step()
                # Stash-restore lại sau repair để arcface retain weights không drift
                model.arcface.weight.data[retain_cls_t] = stashed.to(device)
 
        scheduler.step()
 
        # ── Per-epoch eval ────────────────────────────────────────────────
        n          = max(1, S["steps"])
        ret_acc    = evaluate_accuracy(model, retain_test_loader, device)
        fgt_acc    = evaluate_accuracy(model, forget_loader, device)
        if _per_person:
            proto_sim = compute_person_proto_sim(model, forget_loader, person_lookup,
                                                 person_protos_old, device)
        else:
            cur_proto  = compute_forget_prototype(model, forget_loader, device)  # CPU
            proto_sim  = float(cur_proto.dot(_p_cpu))
        proto_drop = initial_proto_sim - proto_sim
        proto_drop_prev = proto_drop  # state cho get_forget_subphase_weights ở epoch KẾ TIẾP
 
        print(
            f"[MUFAC][{epoch+1:02d}/{total_epochs}][{phase_name}/{subphase_name}]  "
            f"erase={S['erase']/n:.4f}  repel={S['repel']/n:.4f}  kl_uni={S['kl_uni']/n:.4f}  "
            f"div={S['div']/n:.4f}  "
            f"w_erase={w_erase:.3f}  w_repel={w_repel:.3f}  w_kl_uni={w_kl_uni:.3f}  w_div={w_div:.3f}  "
            f"anc={S['anchor']/n:.4f}  proto_r={S['proto']/n:.4f}  ewc={S['ewc']/n:.4f}  "
            f"proto_sim={proto_sim:.4f}(drop:{proto_drop:+.4f})  "
            f"H={S['entropy']/n:.2f}/{math.log(num_classes):.2f}  "
            f"fgt_acc={fgt_acc:.4f}  ret_acc={ret_acc:.4f}  "
            f"conf={S['n_conf']/n:.1f}  boost={boost:.0f}×  "
            f"anchor_w={w_local:.1f}"
        )
 
        # ── Safety guardrail (grace period + consecutive-violation patience) ──
        floor = cfg.retain_acc_floor * retain_acc_ref
        if epoch >= cfg.safety_grace_epochs and ret_acc < floor:
            violation_streak += 1
            print(f"[MUFAC] ⚠ retain_acc dưới floor ({ret_acc:.4f} < {floor:.4f})  "
                  f"streak={violation_streak}/{cfg.safety_patience}")
            if violation_streak >= cfg.safety_patience:
                print(f"[MUFAC] ⚠ SAFETY STOP epoch {epoch+1}: vi phạm floor "
                      f"{cfg.safety_patience} epoch liên tiếp")
                model.load_state_dict(best_state)
                break
        else:
            violation_streak = 0
 
        # Best checkpoint: maximize forget_score × retain_ratio
        retain_score = ret_acc / retain_acc_ref
        proto_forget_score = max(0.0, proto_drop / (1.0 + initial_proto_sim))
        # v_B — TÍN HIỆU MỚI: acc_forget_score đo % quãng đường fgt_acc đã đi
        # từ initial_fgt_acc (Original) tới rc (1/C, mục tiêu Hướng B) — đây
        # CHÍNH LÀ điều L_kl_uni đang cố tạo ra, nhưng trước bản vá này
        # checkpoint selection hoàn toàn không nhìn thấy nó (chỉ có
        # proto_forget_score, thuần representation-level). Kết hợp cả hai để
        # checkpoint cuối cùng thực sự phản ánh hiệu ứng L_kl_uni.
        acc_forget_span  = max(initial_fgt_acc - rc, 1e-6)
        acc_forget_score = min(1.0, max(0.0, (initial_fgt_acc - fgt_acc) / acc_forget_span))
        combined_forget_score = 0.5 * proto_forget_score + 0.5 * acc_forget_score
        # v7.6/v7.9 — mục tiêu gốc: retain_accuracy CAO NHẤT, dùng gate để
        # sau ngưỡng chỉ còn retain_score quyết định (bỏ qua forget hoàn
        # toàn). v_B — SỬA: sau gate, retain vẫn là thành phần CHỦ ĐẠO
        # (trọng số 0.85) nhưng combined_forget_score được giữ lại một phần
        # nhỏ (0.15) làm tiêu chí phụ — phá vỡ thế "chỉ chọn theo retain cao
        # nhất, bỏ qua mọi tín hiệu forget" từng khiến L_kl_uni vô tác dụng
        # với checkpoint cuối cùng dù vẫn chạy trong training. Dưới gate,
        # vẫn cần combined_forget_score>0 một chút để không chọn nhầm epoch 0
        # (chưa unlearn gì cả).
        # v_scaleup (2026-08-17) — tăng trọng số combined_forget_score sau gate
        # từ 0.15 lên 0.3 (ngược lại retain từ 0.85 xuống 0.7) — retain vẫn là
        # thành phần chính nhưng tín hiệu forget (bao gồm acc_forget_score,
        # điều L_kl_uni đang cố tạo ra) giờ có tiếng nói lớn hơn trong việc
        # chọn checkpoint cuối cùng, thay vì chỉ là một tiêu chí phụ yếu ớt.
        gate = cfg.checkpoint_retain_gate
        if retain_score >= gate:
            balance = retain_score * (0.7 + 0.3 * combined_forget_score)
        else:
            balance = max(combined_forget_score, 1e-3) * (retain_score / gate)
 
        if balance > best_balance:
            best_balance = balance
            best_state   = copy.deepcopy(model.state_dict())
            print(f"          ★ New best  balance={best_balance:.4f}  "
                  f"proto_drop={proto_drop:+.4f}  proto_fs={proto_forget_score:.3f}  "
                  f"acc_fs={acc_forget_score:.3f} (fgt_acc={fgt_acc:.4f}→rc={rc:.4f})  "
                  f"retain={retain_score:.3f}")
 
    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad = True
    print(f"\n[MUFAC] Done. Best balance: {best_balance:.4f}")
    return model

# ===========================================================================
# ▓  CELL I2: POST-UNLEARNING RETAIN REPAIR
# ===========================================================================
 
def run_retain_repair(
    model:              nn.Module,
    old_model:          nn.Module,
    retain_loader:      DataLoader,
    retain_test_loader: DataLoader,
    forget_ids:         List,
    num_classes:        int,
    cfg:                Config,
    retain_prototypes:  Optional[torch.Tensor] = None,   # [C, D] frozen, CPU
    retain_val_loader:  Optional[DataLoader]    = None,   # best-ckpt selection
    forget_loader:      Optional[DataLoader]    = None,   # v7.4: guard chống un-erase
    forget_proto:       Optional[torch.Tensor]  = None,   # v7.4: [D] frozen, CPU
    retain_acc_ref:     Optional[float]         = None,   # v7.4: cho R3 catch-up
    beat_acc:           Optional[float]         = None,   # v7.7: retain_accuracy cao nhất
                                                            # trong số baseline ĐÃ CHẠY (FineTune/
                                                            # NegGrad) — R3 nhắm target là
                                                            # max(retain_acc_ref, beat_acc).
) -> nn.Module:
    """
    Two-phase post-unlearning retain repair.
 
    Phase R1 (60% epochs, high LR): CE dominant, neck + retain-arcface +
      layer4 (lr nhỏ hơn) cùng adapt với embedding space mới.
    Phase R2 (40% epochs, low LR): KD-stabilisation dominant, tránh
      over-drift khỏi representation gốc.
 
    Best-checkpoint selection: theo dõi best state trên retain_val_loader
    (KHÔNG dùng retain_test_loader để chọn — tránh leak test set vào lựa
    chọn checkpoint), rồi load lại best_state trước khi trả về model.
    """
    device = cfg.device
    best_val_acc, best_state = -1.0, None
    model.to(device)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad = False
 
    # MUFAC: forget_ids là family-id (str), KHÔNG map được sang
    # torch.tensor(..., dtype=torch.long) như VGGFace2/Korean Family (sẽ
    # crash). Về ý nghĩa: không có "hàng ArcFace của gia đình bị quên" nào
    # tồn tại riêng biệt để đóng băng (1 family trải trên nhiều age-class,
    # dùng chung hàng ArcFace với các family retain cùng tuổi) — nên bỏ qua
    # hẳn cơ chế stash/freeze forget-ArcFace-row.
    forget_cls_t = None
    stashed_forget_arcface = None
    if identity_equals_class(forget_ids, model.arcface.weight.shape[0]):
        fset = set(forget_ids)
        forget_cls_t = torch.tensor(list(fset), dtype=torch.long)
        # Stash forget arcface weights (state after unlearning — NOT original)
        stashed_forget_arcface = model.arcface.weight.data[forget_cls_t].clone()
 
    # Retain-prototype anchor cũng áp dụng trong bước repair — cùng ý tưởng
    # "neo về prototype đúng lớp" như trong RetainLossV7.
    proto_dev = (
        retain_prototypes.to(device)
        if (cfg.use_proto_retain and retain_prototypes is not None) else None
    )
 
    # v7.4: guard chống un-erase — forget_iter độc lập với retain_loader,
    # lặp vòng qua forget_loader song song trong cả R1 và R2.
    guard_w = cfg.repair_forget_guard_weight
    forget_proto_dev = None
    forget_iter = None
    if forget_loader is not None and forget_proto is not None and guard_w > 0:
        forget_proto_dev = forget_proto.to(device)
        forget_iter = iter(forget_loader)
 
    def _guard_loss(model):
        """Lực nhỏ tiếp tục đẩy embedding forget ra xa old_forget_proto,
        song song với CE/KD trên retain — ngăn repair kéo neck/layer4
        trôi ngược về gần cấu hình gốc."""
        nonlocal forget_iter
        if forget_iter is None:
            return torch.zeros(1, device=device).squeeze()
        try:
            x_f, _, _ = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            x_f, _, _ = next(forget_iter)
        x_f = x_f.to(device)
        z_f, _ = model(x_f, None)
        return guard_w * (z_f * forget_proto_dev.unsqueeze(0)).sum(dim=1).mean()
 
    T  = cfg.kd_temperature
    ce = nn.CrossEntropyLoss()
 
    r1_epochs = max(1, int(cfg.repair_epochs * 0.6))
    r2_epochs = cfg.repair_epochs - r1_epochs
 
    print(f"\n[Repair] Two-phase retain repair — {cfg.repair_epochs} epochs total")
    print(f"         R1={r1_epochs}e (LR={cfg.repair_lr:.1e}, CE+KD, neck+retain-arcface adapt, "
          f"layer4 LR={cfg.repair_lr*cfg.repair_layer4_lr_mult:.1e})")
    print(f"         R2={r2_epochs}e (LR={cfg.repair_lr*0.1:.1e}, KD stabilisation)")
    print(f"         KEY: only FORGET arcface frozen (nếu identity==class); retain arcface + layer4 free to adapt")
 
    # ── PHASE R1: Adaptive neck + retain-arcface + layer4 re-alignment ──────
    trainable_r1 = [p for n, p in model.named_parameters()
                    if ("neck" in n or "arcface" in n) and "backbone.layer4" not in n]
    layer4_r1    = [p for n, p in model.named_parameters() if "backbone.layer4" in n]
    all_r1_params = trainable_r1 + layer4_r1
    opt_r1 = torch.optim.AdamW(
        [
            {"params": trainable_r1, "lr": cfg.repair_lr},
            {"params": layer4_r1,    "lr": cfg.repair_lr * cfg.repair_layer4_lr_mult},
        ],
        weight_decay=1e-5
    )
    sched_r1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_r1, T_max=max(1, r1_epochs), eta_min=cfg.repair_lr * 0.1
    )
 
    for epoch in range(r1_epochs):
        model.train()
        total_loss = 0.0
        for x_r, y_r, _ in retain_loader:
            x_r, y_r = x_r.to(device), y_r.to(device)
 
            z_r, logits_new = model(x_r, y_r)
 
            # CE: direct accuracy objective — dominant in R1
            loss_ce = cfg.w_ce_retain * ce(logits_new, y_r)
 
            # Light KD: soft guard against catastrophic over-fitting
            with torch.no_grad():
                _, logits_old = old_model(x_r, y_r)
            log_p = F.log_softmax(logits_new / T, dim=1)
            p_old = F.softmax(logits_old   / T, dim=1)
            loss_kl = 5.0 * (T ** 2) * F.kl_div(log_p, p_old, reduction='batchmean')
 
            # Retain-prototype anchor — kéo embedding về đúng prototype gốc
            if proto_dev is not None:
                proto_y = proto_dev[y_r]
                loss_proto = cfg.w_proto_retain * (1.0 - (z_r * proto_y).sum(dim=1)).mean()
            else:
                loss_proto = torch.zeros(1, device=device).squeeze()
 
            loss_guard = _guard_loss(model)
 
            loss = loss_ce + loss_kl + loss_proto + loss_guard
 
            opt_r1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_r1_params, 1.0)
            opt_r1.step()
 
            # Only restore FORGET arcface (retain arcface adapts freely)
            if forget_cls_t is not None:
                model.arcface.weight.data[forget_cls_t] = stashed_forget_arcface.to(device)
 
            total_loss += float(loss)
 
        sched_r1.step()
        val_source = retain_val_loader if retain_val_loader is not None else retain_test_loader
        ret_acc = evaluate_accuracy(model, val_source, device)
        if ret_acc > best_val_acc:
            best_val_acc = ret_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"[R1][{epoch+1:02d}/{r1_epochs}]  "
              f"loss={total_loss/len(retain_loader):.4f}  "
              f"retain_val_acc={ret_acc:.4f}  best={best_val_acc:.4f}")
 
    # ── PHASE R2: KD Stabilisation ────────────────────────────────────────────
    trainable_r2 = [p for n, p in model.named_parameters()
                    if ("neck" in n or "arcface" in n) and "backbone.layer4" not in n]
    layer4_r2    = [p for n, p in model.named_parameters() if "backbone.layer4" in n]
    all_r2_params = trainable_r2 + layer4_r2
    opt_r2 = torch.optim.AdamW(
        [
            {"params": trainable_r2, "lr": cfg.repair_lr * 0.1},
            {"params": layer4_r2,    "lr": cfg.repair_lr * 0.1 * cfg.repair_layer4_lr_mult},
        ],
        weight_decay=1e-5
    )
 
    for epoch in range(r2_epochs):
        model.train()
        total_loss = 0.0
        for x_r, y_r, _ in retain_loader:
            x_r, y_r = x_r.to(device), y_r.to(device)
 
            z_r, logits_new = model(x_r, y_r)
            with torch.no_grad():
                _, logits_old = old_model(x_r, y_r)
 
            # KD dominant in R2 — stabilise without over-drifting.
            # repair_r2_kd_mult cho phép hạ lực KD riêng cho dataset khó.
            log_p = F.log_softmax(logits_new / T, dim=1)
            p_old = F.softmax(logits_old   / T, dim=1)
            loss_kl = (cfg.w_kd_global * cfg.repair_r2_kd_mult
                       * (T ** 2) * F.kl_div(log_p, p_old, reduction='batchmean'))
 
            loss_ce = cfg.w_ce_retain * ce(logits_new, y_r)
 
            if proto_dev is not None:
                proto_y = proto_dev[y_r]
                loss_proto = (cfg.w_proto_retain * 0.5) * (1.0 - (z_r * proto_y).sum(dim=1)).mean()
            else:
                loss_proto = torch.zeros(1, device=device).squeeze()
 
            loss_guard = _guard_loss(model)
 
            loss = loss_kl + loss_ce + loss_proto + loss_guard
 
            opt_r2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_r2_params, cfg.grad_clip_retain)
            opt_r2.step()
 
            # Keep forget arcface frozen
            if forget_cls_t is not None:
                model.arcface.weight.data[forget_cls_t] = stashed_forget_arcface.to(device)
 
            total_loss += float(loss)
 
        val_source = retain_val_loader if retain_val_loader is not None else retain_test_loader
        ret_acc = evaluate_accuracy(model, val_source, device)
        if ret_acc > best_val_acc:
            best_val_acc = ret_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"[R2][{epoch+1:02d}/{r2_epochs}]  "
              f"loss={total_loss/len(retain_loader):.4f}  "
              f"retain_val_acc={ret_acc:.4f}  best={best_val_acc:.4f}")
 
    # ── PHASE R3: CATCH-UP (v7.4) ────────────────────────────────────────
    # Sau R1+R2, nếu retain_val_acc vẫn thấp hơn tham chiếu (retain_acc_ref,
    # tức Original), chủ động chạy thêm epoch CE-dominant (vẫn kèm guard để
    # không xoá ngược phần đã erase) cho tới khi bắt kịp target hoặc chạm
    # trần repair_catchup_max_epochs. Đây là cơ chế chủ động thay vì trông
    # chờ vào may rủi giữa các lần chạy — nhưng KHÔNG đảm bảo tuyệt đối đạt
    # target nếu guard quá mạnh so với những gì CE có thể kéo lại được;
    # trần epoch đảm bảo vòng lặp luôn dừng với chi phí giới hạn.
    if cfg.repair_catchup_enable and retain_acc_ref is not None:
        # v7.7 KHẨN — target = MAX(retain_acc_ref, beat_acc) × catchup_target,
        # đảm bảo LPEU nhắm đúng con số cần vượt trong bảng kết quả cuối.
        base_target = max(retain_acc_ref, beat_acc) if beat_acc is not None else retain_acc_ref
        target_acc = base_target * cfg.repair_catchup_target
        if best_val_acc < target_acc:
            print(f"\n[Repair] R3 catch-up: best_val_acc={best_val_acc:.4f} "
                  f"< target={target_acc:.4f} (retain_acc_ref×{cfg.repair_catchup_target}) "
                  f"— chạy thêm tối đa {cfg.repair_catchup_max_epochs} epoch")
 
            trainable_r3 = [p for n, p in model.named_parameters()
                             if ("neck" in n or "arcface" in n) and "backbone.layer4" not in n]
            layer4_r3     = [p for n, p in model.named_parameters() if "backbone.layer4" in n]
            all_r3_params = trainable_r3 + layer4_r3
            opt_r3 = torch.optim.AdamW(
                [
                    {"params": trainable_r3, "lr": cfg.repair_lr * 0.5},
                    {"params": layer4_r3,    "lr": cfg.repair_lr * 0.5 * cfg.repair_layer4_lr_mult},
                ],
                weight_decay=1e-5
            )
 
            catchup_epoch = 0
            while best_val_acc < target_acc and catchup_epoch < cfg.repair_catchup_max_epochs:
                catchup_epoch += 1
                model.train()
                total_loss = 0.0
                for x_r, y_r, _ in retain_loader:
                    x_r, y_r = x_r.to(device), y_r.to(device)
                    z_r, logits_new = model(x_r, y_r)
 
                    loss_ce = cfg.w_ce_retain * ce(logits_new, y_r)
 
                    with torch.no_grad():
                        _, logits_old = old_model(x_r, y_r)
                    log_p = F.log_softmax(logits_new / T, dim=1)
                    p_old = F.softmax(logits_old   / T, dim=1)
                    loss_kl = 5.0 * (T ** 2) * F.kl_div(log_p, p_old, reduction='batchmean')
 
                    if proto_dev is not None:
                        proto_y = proto_dev[y_r]
                        loss_proto = cfg.w_proto_retain * (1.0 - (z_r * proto_y).sum(dim=1)).mean()
                    else:
                        loss_proto = torch.zeros(1, device=device).squeeze()
 
                    loss_guard = _guard_loss(model)
 
                    loss = loss_ce + loss_kl + loss_proto + loss_guard
 
                    opt_r3.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_r3_params, 1.0)
                    opt_r3.step()
 
                    if forget_cls_t is not None:
                        model.arcface.weight.data[forget_cls_t] = stashed_forget_arcface.to(device)
 
                    total_loss += float(loss)
 
                val_source = retain_val_loader if retain_val_loader is not None else retain_test_loader
                ret_acc = evaluate_accuracy(model, val_source, device)
                if ret_acc > best_val_acc:
                    best_val_acc = ret_acc
                    best_state = copy.deepcopy(model.state_dict())
                print(f"[R3][{catchup_epoch:02d}/{cfg.repair_catchup_max_epochs}]  "
                      f"loss={total_loss/len(retain_loader):.4f}  "
                      f"retain_val_acc={ret_acc:.4f}  best={best_val_acc:.4f}  target={target_acc:.4f}")
 
            if best_val_acc >= target_acc:
                print(f"[Repair] R3: đã bắt kịp target ({best_val_acc:.4f} >= {target_acc:.4f})")
            else:
                print(f"[Repair] R3: chạm trần {cfg.repair_catchup_max_epochs} epoch, "
                      f"best_val_acc={best_val_acc:.4f} vẫn < target={target_acc:.4f}")
 
    # Khôi phục checkpoint TỐT NHẤT theo retain_val_loader, không phải state
    # của epoch cuối cùng.
    if best_state is not None:
        model.load_state_dict(best_state)
    final_acc = evaluate_accuracy(model, retain_test_loader, device)
    print(f"\n[Repair] Done. Best retain_val_acc={best_val_acc:.4f} → "
          f"Final retain_TEST_acc={final_acc:.4f}")
    return model

# ===========================================================================
# ▓  CELL J: EVALUATION METRICS
# ===========================================================================

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval(); correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x, None)
        correct += int((logits.argmax(1) == y).sum())
        total   += y.numel()
    return correct / max(1, total)

@torch.no_grad()
def mean_sim_to_proto(model, loader, proto, device):
    model.eval(); sims = []
    for x, _, _ in loader:
        z, _ = model(x.to(device), None)
        sims.append((z @ proto.to(device)).mean().item())
    return float(np.mean(sims)) if sims else 0.0

@torch.no_grad()
def mean_entropy(model, loader, device):
    model.eval(); vals = []
    for x, _, _ in loader:
        _, logits = model(x.to(device), None)
        p = F.softmax(logits, dim=1)
        vals.append(-(p * torch.log(p + 1e-8)).sum(1).mean().item())
    return float(np.mean(vals)) if vals else 0.0

@torch.no_grad()
def cluster_compactness(model, loader, device):
    model.eval()
    embs = [model(x.to(device), None)[0].cpu() for x, _, _ in loader]
    z = torch.cat(embs)
    return float(((z - z.mean(0, keepdim=True))**2).sum(1).mean())

@torch.no_grad()
def intra_cluster_cos_sim(model, loader, device):
    model.eval()
    embs = [model(x.to(device), None)[0].cpu() for x, _, _ in loader]
    z = F.normalize(torch.cat(embs), dim=1)
    if z.size(0) < 2: return 1.0
    sim_mat = z @ z.T
    mask = ~torch.eye(z.size(0), dtype=torch.bool)
    return float(sim_mat[mask].mean())

@torch.no_grad()
def neighbor_shift(new_m, old_m, loader, device):
    new_m.eval(); old_m.eval(); shifts = []
    for x, _, _ in loader:
        x = x.to(device)
        z_new, _ = new_m(x, None); z_old, _ = old_m(x, None)
        shifts.append((z_new - z_old).norm(dim=1).mean().item())
    return float(np.mean(shifts)) if shifts else 0.0

@torch.no_grad()
def get_max_confidence(model, loader, device, n_max=500):
    """Giữ lại để tương thích ngược — KHÔNG dùng cho MIA (xem get_loss_signal)."""
    model.eval(); scores = []
    for x, _, _ in loader:
        _, logits = model(x.to(device), None)
        scores.append(F.softmax(logits, dim=1).max(1).values.cpu())
        if sum(len(s) for s in scores) >= n_max: break
    return torch.cat(scores)[:n_max]

def get_loss_signal(model, loader, device, n_max=500):
    """
    MIA feature = per-sample CE loss (Yeom et al. 2018 — loss-based
    membership inference). KHÔNG dùng max-softmax-confidence: với
    arcface_s=32.0, softmax(logits*s) gần như luôn ~0.999+ một khi model
    hội tụ tốt (dù đúng hay sai) → mất hết phương sai phân biệt → AUC rơi
    về đúng 0.5 (bão hòa, không phải model đạt privacy hoàn hảo).
    """
    model.eval(); scores = []
    ce = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            _, logits = model(x, None)
            loss = ce(logits, y)
            scores.append((-loss).cpu())   # loss thấp (nhớ tốt) → score cao → "member"
            if sum(len(s) for s in scores) >= n_max: break
    return torch.cat(scores)[:n_max]

def _fit_mia_attacker(model, member_loader, nonmember_loader, device, cfg, n_bootstrap=30):
    """v7.9-fix5 — cân bằng lớp member/non-member (cắt lớp nhiều hơn xuống
    bằng lớp ít hơn) TRƯỚC khi fit — sửa bug mất cân bằng lớp khiến
    accuracy suy biến thành "luôn đoán lớp đa số" (xem lịch sử: từng ra
    accuracy GIỐNG HỆT nhau ở mọi model do bug này).

    v7.9-fix6 — BẰNG CHỨNG THỰC NGHIỆM (2026-08-16, sau fix5): với
    n_per_class=55 (khá nhỏ vì forget-unit là cá nhân, chỉ ~20 người),
    sai số chuẩn của 1 ước lượng accuracy đơn lẻ ≈ sqrt(0.5×0.5/55) ≈ 0.067
    → khoảng tin cậy 95% rộng tới ±13 điểm phần trăm. Chênh lệch giữa các
    phương pháp trong 1 lần chạy (thường chỉ 3-5 điểm) NẰM GỌN trong biên độ
    nhiễu này — 1 điểm ước lượng (point estimate) đơn lẻ KHÔNG đủ để kết
    luận phương pháp nào thật sự hơn. Thêm bootstrap resampling (mặc định
    30 lần, resample có hoàn lại từ chính k mẫu/lớp đã thu thập, fit lại
    logistic regression mỗi lần) để có độ lệch chuẩn đi kèm — nếu chênh
    lệch giữa 2 phương pháp nhỏ hơn ~1-2 lần std cộng lại, nên coi là KHÔNG
    phân biệt được về mặt thống kê, không phải "phương pháp A tốt hơn B".

    Trả về (auc, accuracy, auc_std, accuracy_std, n_per_class). auc/accuracy
    = None nếu không đủ dữ liệu để fit (n_per_class < 1 hoặc chỉ có 1 lớp).
    auc_std/accuracy_std = 0.0 nếu n_per_class < 2 (không đủ để bootstrap)."""
    n   = min(cfg.mia_n_members, cfg.mia_n_nonmembers)
    mem = get_loss_signal(model, member_loader,    device, n)
    non = get_loss_signal(model, nonmember_loader, device, n)
    k = min(len(mem), len(non))
    if k < 1:
        return None, None, None, None, 0
    mem_np = mem[:k].numpy()
    non_np = non[:k].numpy()

    def _fit_once(mem_s, non_s):
        X = np.concatenate([mem_s, non_s]).reshape(-1, 1)
        y = np.array([1] * len(mem_s) + [0] * len(non_s))
        if len(np.unique(y)) < 2:
            return None, None
        try:
            clf = LogisticRegression(max_iter=500)
            clf.fit(X, y)
            auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
            acc = float(clf.score(X, y))
            return auc, acc
        except Exception:
            return None, None

    auc0, acc0 = _fit_once(mem_np, non_np)   # point estimate — fit trên đủ k mẫu/lớp
    if auc0 is None:
        return None, None, None, None, k
    if k < 2 or n_bootstrap <= 0:
        return auc0, acc0, 0.0, 0.0, k

    rng = np.random.RandomState(cfg.seed)
    aucs, accs = [], []
    for _ in range(n_bootstrap):
        mem_s = mem_np[rng.randint(0, k, size=k)]   # resample có hoàn lại
        non_s = non_np[rng.randint(0, k, size=k)]
        a, c = _fit_once(mem_s, non_s)
        if a is not None:
            aucs.append(a); accs.append(c)
    auc_std = float(np.std(aucs)) if aucs else 0.0
    acc_std = float(np.std(accs)) if accs else 0.0
    return auc0, acc0, auc_std, acc_std, k


def compute_mia_auc(model, forget_loader, retain_test_loader, device, cfg):
    auc, _, _, _, _ = _fit_mia_attacker(model, forget_loader, retain_test_loader, device, cfg)
    return auc if auc is not None else 0.5


def compute_mia_auc_unseen(model, forget_loader, unseen_loader, device, cfg):
    """v7.7 — Forgetting Score đúng định nghĩa Choi & Na (2023) / GLI (Choi
    et al. 2024): so loss của x_forget với loss của x_unseen (hộ gia đình /
    cá nhân CHƯA TỪNG được model gốc nhìn thấy), KHÔNG phải x_retain_test.
    Trả về AUC — giữ lại để tham khảo, KHÔNG dùng để so trực tiếp với
    Forgetting Score trong 2 bài báo (paper dùng accuracy — xem
    compute_mia_accuracy_unseen)."""
    if unseen_loader is None:
        return None
    auc, _, _, _, _ = _fit_mia_attacker(model, forget_loader, unseen_loader, device, cfg)
    return auc if auc is not None else 0.5


def compute_mia_accuracy_unseen(model, forget_loader, unseen_loader, device, cfg):
    """v7.9-fix4 — ĐÚNG 100% công thức Forgetting Score trong paper gốc.

    Trích Choi & Na (2023): "M denotes the accuracy of ψ(·)". Trích GLI:
    "forgetting score as abs(0.5 − M) where M denotes the ACCURACY of the
    MIA model" (không phải AUC). Lớp member/non-member được CÂN BẰNG trước
    khi fit — xem _fit_mia_attacker (v7.9-fix5/fix6)."""
    if unseen_loader is None:
        return None
    _, acc, _, _, _ = _fit_mia_attacker(model, forget_loader, unseen_loader, device, cfg)
    return acc if acc is not None else 0.5


def full_evaluate(
    name: str,
    model: nn.Module,
    old_model: nn.Module,
    retain_test_loader: DataLoader,
    forget_eval_loader: DataLoader,
    local_loader: DataLoader,
    old_proto: torch.Tensor,
    cfg: Config,
    num_classes: int,
    unseen_loader: Optional[DataLoader] = None,   # v7.7 — cho Forgetting Score đúng chuẩn Choi/GLI
) -> Dict:
    device = cfg.device
    rc     = 1.0 / num_classes

    ret_acc  = evaluate_accuracy(model, retain_test_loader, device)
    fgt_acc  = evaluate_accuracy(model, forget_eval_loader, device)
    old_sim  = mean_sim_to_proto(old_model, forget_eval_loader, old_proto, device)
    new_sim  = mean_sim_to_proto(model,     forget_eval_loader, old_proto, device)
    nshift   = neighbor_shift(model, old_model, local_loader, device)
    old_ent  = mean_entropy(old_model, forget_eval_loader, device)
    new_ent  = mean_entropy(model,     forget_eval_loader, device)
    old_comp = cluster_compactness(old_model, forget_eval_loader, device)
    new_comp = cluster_compactness(model,     forget_eval_loader, device)
    old_ics  = intra_cluster_cos_sim(old_model, forget_eval_loader, device)
    new_ics  = intra_cluster_cos_sim(model,     forget_eval_loader, device)
    mia      = compute_mia_auc(model, forget_eval_loader, retain_test_loader, device, cfg)
    # v7.9-fix5/fix6 — fit MỘT LẦN cho cặp (forget, unseen), lấy AUC, accuracy,
    # và std (từ bootstrap) từ cùng 1 lượt gọi — xem _fit_mia_attacker.
    _mia_auc_u, _mia_acc_u, _mia_auc_std, _mia_acc_std, _mia_n_per_class = _fit_mia_attacker(
        model, forget_eval_loader, unseen_loader, device, cfg
    ) if unseen_loader is not None else (None, None, None, None, 0)
    mia_unseen     = _mia_auc_u  # v7.7 — AUC (tham khảo, KHÔNG khớp paper)
    mia_acc_unseen = _mia_acc_u  # v7.9-fix4 — accuracy, ĐÚNG paper

    m = dict(
        retain_accuracy       = ret_acc,
        forget_accuracy       = fgt_acc,
        forget_acc_vs_random  = fgt_acc - rc,
        old_forget_sim        = old_sim,
        new_forget_sim        = new_sim,
        forget_sim_drop       = old_sim - new_sim,
        neighbor_shift        = nshift,
        old_entropy           = old_ent,
        new_entropy           = new_ent,
        entropy_increase      = new_ent - old_ent,
        old_compactness       = old_comp,
        new_compactness       = new_comp,
        compactness_increase  = new_comp - old_comp,
        old_intra_cluster_sim = old_ics,
        new_intra_cluster_sim = new_ics,
        cluster_sim_drop      = old_ics - new_ics,
        mia_auc               = mia,
        mia_auc_unseen        = mia_unseen,  # v7.7 — None nếu không truyền unseen_loader
        # v7.9-fix3 — Forgetting Score ĐÚNG công thức Choi & Na (2023) / GLI
        # (Choi et al. 2024): |0.5 - M|, M = AUC bộ phân loại MIA. CÀNG THẤP
        # CÀNG TỐT (0 = hoàn hảo, kẻ tấn công không phân biệt được
        # forget/unseen). forgetting_score_unseen dùng mia_auc_unseen (ĐÚNG
        # chuẩn Choi/GLI — so forget với ảnh chưa từng thấy) — đây là con số
        # nên dùng để so LPEU với GLI/FineTune/NegGrad. forgetting_score (vs
        # retain) giữ lại để tham khảo, KHÔNG so trực tiếp với Choi/GLI.
        forgetting_score        = abs(mia - 0.5),
        forgetting_score_unseen = abs(mia_unseen - 0.5) if mia_unseen is not None else None,       # v7.7 — dựa AUC, chỉ tham khảo
        # v7.9-fix4 — con số DÙNG ĐỂ SO VỚI PAPER: M = accuracy (không phải AUC).
        mia_accuracy_unseen           = mia_acc_unseen,
        forgetting_score_unseen_paper = abs(mia_acc_unseen - 0.5) if mia_acc_unseen is not None else None,
        mia_n_per_class                = _mia_n_per_class,  # v7.9-fix5 — số mẫu MỖI lớp thực dùng để fit attacker (forget vs unseen)
        # v7.9-fix6 — độ lệch chuẩn (bootstrap, 30 lần resample) của accuracy.
        # forgetting_score_unseen_paper_std XẤP XỈ std của |acc-0.5| bằng std
        # của acc (đúng khi acc không quá sát 0 hoặc 1) — dùng để đánh giá 2
        # phương pháp có KHÁC NHAU CÓ Ý NGHĨA THỐNG KÊ hay chỉ là nhiễu do
        # n_per_class nhỏ. Quy tắc thô: nếu |FS_A - FS_B| < (std_A + std_B),
        # coi là KHÔNG phân biệt được.
        mia_accuracy_unseen_std        = _mia_acc_std,
        forgetting_score_unseen_paper_std = _mia_acc_std,
    )

    bar = "═" * 70
    tgt_fgt = "✓" if abs(fgt_acc - rc) < 0.01 else ("⚠ low"  if fgt_acc < rc*0.5 else "⚠ high")
    tgt_sim = "✓" if (old_sim - new_sim) > 0.25  else "·"
    tgt_ns  = "✓" if nshift < 0.15               else "⚠"
    tgt_ent = "✓" if (new_ent - old_ent) > 0.5   else "·"
    tgt_mia = "✓" if abs(mia - 0.5) < 0.03       else "·"
    tgt_ics = "✓" if (old_ics - new_ics) > 0.20  else "·"

    print(f"\n{bar}")
    print(f"  [{name}]")
    print(f"{bar}")
    print(f"  Retain Accuracy      : {ret_acc:.4f}          ← HIGH is good")
    print(f"  Forget Accuracy      : {fgt_acc:.4f}  {tgt_fgt}  ← ~{rc:.4f} ideal (1/C)")
    print(f"  Forget Sim Drop  ↑   : {old_sim-new_sim:+.4f}  {tgt_sim}  ← ≥ +0.25 target")
    print(f"  Cluster Cos Drop ↑   : {old_ics-new_ics:+.4f}  {tgt_ics}  ← ≥ +0.20 target")
    print(f"  Compactness Incr ↑   : {new_comp-old_comp:+.4f}          ← positive = dispersed")
    print(f"  Entropy Increase ↑   : {new_ent-old_ent:+.4f}  {tgt_ent}  ← ≥ +0.50 target")
    print(f"  Neighbor Shift   ↓   : {nshift:.4f}  {tgt_ns}  ← < 0.15 target")
    print(f"  MIA AUC (vs retain)  : {mia:.4f}  {tgt_mia}  ← 0.5 ideal (tín hiệu forget-vs-retain, KHÔNG so được với Choi/GLI)")
    if mia_unseen is not None:
        tgt_mia_u = "✓" if abs(mia_unseen - 0.5) < 0.03 else "·"
        print(f"  MIA AUC (vs unseen)  : {mia_unseen:.4f}  {tgt_mia_u}  ← 0.5 ideal — ĐÚNG chuẩn Forgetting Score Choi & Na / GLI")
        print(f"  ┌─ Forgetting Score (AUC-based, chỉ tham khảo)  : {abs(mia_unseen-0.5):.4f}")
        tgt_mia_acc = "✓" if abs(mia_acc_unseen - 0.5) < 0.03 else "·"
        print(f"  MIA Accuracy (vs unseen)  : {mia_acc_unseen:.4f}  {tgt_mia_acc}  ← 0.5 ideal — ĐÚNG M trong paper (accuracy, không phải AUC)")
        print(f"  └─ FORGETTING SCORE (đúng công thức paper: |0.5-accuracy|) : {abs(mia_acc_unseen-0.5):.4f}  ← DÙNG SỐ NÀY để so với GLI/Choi&Na")
        print(f"     (n={_mia_n_per_class} mẫu/lớp, đã cân bằng — v7.9-fix5. Nếu n rất nhỏ (<15), "
              f"kết quả accuracy dễ nhiễu — cân nhắc tăng num_forget_ids hoặc đọc thêm forgetting_score_unseen làm tham khảo chéo)")
        print(f"     ± {_mia_acc_std:.4f} (std, bootstrap {30 if _mia_n_per_class >= 2 else 0} lần) — v7.9-fix6. "
              f"So 2 phương pháp: nếu |chênh lệch FS| < tổng 2 std, coi là KHÔNG khác biệt có ý nghĩa thống kê.")
    else:
        print(f"  MIA AUC (vs unseen)  : (chưa truyền unseen_loader)")
    print(f"{bar}")
    return m

# ===========================================================================
# ▓  CELL J2: RE-IDENTIFICATION LEAKAGE METRIC (embedding retrieval attack)
# ===========================================================================
#
# forget_accuracy (Cell J) đo ở CẤP TASK (phân loại tuổi) — với MUFAC, person
# identity KHÔNG phải là 1 trong 8 class tuổi, nên đầu ArcFace luôn được đóng
# băng/khôi phục sau mỗi bước forget (identity_equals_class=False). Kết quả:
# forget_accuracy không thể giảm dù w_kl_uniform tăng bao nhiêu — đây là tính
# chất KIẾN TRÚC, không phải bug (xem companion paper, Section "Discussion").
#
# Metric dưới đây đo "quên" ở ĐÚNG CẤP mà LPEU thực sự tác động — hình học
# embedding — và mô phỏng trực tiếp threat model mở đầu bài báo LPEU: một kẻ
# tấn công có quyền truy cập embedding (không cần đầu phân loại) tìm kiếm
# hàng-xóm-gần-nhất (nearest-neighbour retrieval) để NHẬN DIỆN LẠI người đã
# bị "quên", giống hệ thống face-retrieval thật. Không đụng tới ArcFace head
# nên KHÔNG bị chặn bởi cơ chế đóng băng nói trên.
#
# v_reid2 — THÊM RANK DIAGNOSTIC (không chỉ top-1 accuracy). Lần chạy đầu ra
# reid_accuracy = 0.0000 cho CẢ 4 phương pháp (kể cả Original chưa hề bị
# unlearn) trên cả 2 file — xác suất điều đó xảy ra thuần do may rủi (nếu
# random đều 1/809) chỉ ~0.05%, nên cần thêm rank để phân biệt 2 khả năng:
#   (a) collapse thật — embedding của các cá nhân KHÁC NHAU gần như không
#       phân biệt được (intra_cluster_sim đo ở Cell J = 0.98, vượt xa ngưỡng
#       cảnh báo 0.80 sẵn có trong "Model Health Check" ở Cell L) → rank
#       đúng của người bị quên sẽ gần với rank NGẪU NHIÊN (~(N+1)/2).
#   (b) có tín hiệu identity thật nhưng luôn "suýt thua" hạng nhất → rank
#       đúng sẽ THẤP HƠN HẲN mức ngẫu nhiên (top-5/top-10 sẽ khác 0 rõ rệt)
#       dù top-1 vẫn bằng 0.
# reid_mean_rank/reid_median_rank càng GẦN reid_chance_mean_rank càng ủng hộ
# giả thuyết (a); càng THẤP hơn nhiều thì ủng hộ (b).

@torch.no_grad()
def compute_reidentification_accuracy(
    model: nn.Module,
    eval_base,
    forget_person_to_idx: Dict[str, List[int]],
    retain_person_to_idx: Dict[str, List[int]],
    cfg: Config,
) -> Dict:
    device = cfg.device
    model.eval()

    def _embed_indices(idx_list):
        if not idx_list:
            return torch.empty(0, cfg.embedding_dim)
        ds = IndexedSubset(eval_base, idx_list)
        loader = make_loader(ds, cfg.batch_size, False, cfg.num_workers)
        embs = []
        for x, _, _ in loader:
            z, _ = model(x.to(device), None)
            embs.append(z.cpu())
        return torch.cat(embs)

    labels, protos, forget_embs = [], [], {}
    for pkey, idxs in retain_person_to_idx.items():
        if not idxs:
            continue
        e = _embed_indices(idxs)
        protos.append(F.normalize(e.mean(0, keepdim=True), dim=1))
        labels.append(pkey)
    for pkey, idxs in forget_person_to_idx.items():
        if not idxs:
            continue
        e = _embed_indices(idxs)
        forget_embs[pkey] = e
        protos.append(F.normalize(e.mean(0, keepdim=True), dim=1))
        labels.append(pkey)

    gallery = torch.cat(protos, dim=0)                          # [N, D]
    N = gallery.size(0)
    label_to_row = {lbl: i for i, lbl in enumerate(labels)}      # 1 identity = 1 hàng

    n_skipped = 0
    ranks = []                        # hạng (1-indexed) của đúng identity, theo similarity giảm dần
    top1 = top5 = top10 = 0
    total = 0
    for pkey, e_all in forget_embs.items():
        k = e_all.size(0)
        if k < 2:
            n_skipped += 1
            continue
        self_row = label_to_row[pkey]
        for i in range(k):
            mask = torch.ones(k, dtype=torch.bool); mask[i] = False
            loo_proto = F.normalize(e_all[mask].mean(0, keepdim=True), dim=1)   # [1, D]
            g = gallery.clone()
            g[self_row] = loo_proto.squeeze(0)          # chỉ thay đúng 1 hàng — không rò rỉ x_i
            query = F.normalize(e_all[i:i+1], dim=1)
            sims = (query @ g.T).squeeze(0)              # [N]
            order = torch.argsort(sims, descending=True)
            rank = int((order == self_row).nonzero(as_tuple=True)[0].item()) + 1
            ranks.append(rank)
            total += 1
            if rank == 1:  top1  += 1
            if rank <= 5:  top5  += 1
            if rank <= 10: top10 += 1

    ranks_t = torch.tensor(ranks, dtype=torch.float32) if ranks else torch.empty(0)
    return dict(
        reid_accuracy                       = top1 / max(1, total),
        reid_top5_accuracy                  = top5 / max(1, total),
        reid_top10_accuracy                 = top10 / max(1, total),
        reid_mean_rank                      = float(ranks_t.mean()) if len(ranks) else None,
        reid_median_rank                    = float(ranks_t.median()) if len(ranks) else None,
        reid_chance_mean_rank               = (N + 1) / 2.0,
        reid_n_queries                      = total,
        reid_gallery_size                   = N,
        reid_n_persons_evaluated            = len(forget_embs) - n_skipped,
        reid_n_persons_skipped_single_image = n_skipped,
    )


# ===========================================================================
# ▓  CELL K: BASELINES — nguồn tham khảo (không tự phát minh)
# ===========================================================================
#
# [1] Fine-tuning-on-retain: Golatkar, Achille, Soatto. "Eternal Sunshine of
#     the Spotless Net: Selective Forgetting in Deep Networks." CVPR 2020.
#     arXiv:1911.04933.
# [2] NegGrad / NegGrad+ (gradient ascent trên forget set): cùng nguồn trên;
#     công thức joint loss (loss_r − loss_f) formal hoá trong Kurmanji et al.
#     (SCRUB), NeurIPS 2023. arXiv:2302.09880.
# [3] Choi, Choi, Lee, Seo, Na. "Towards Efficient Machine Unlearning with
#     Data Augmentation: Guided Loss-Increasing (GLI)." CVPR Workshops 2024
#     — cùng nhóm tác giả với MUFAC (Dongbin Na, POSTECH), cùng dùng
#     Fine-tuning + NegGrad làm baseline.
# ===========================================================================

def run_finetune(model, retain_loader, cfg, num_epochs=5):
    """Baseline: Fine-tuning-on-retain."""
    device = cfg.device; model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=getattr(cfg, 'finetune_lr', 1e-4), momentum=0.9, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()
    for _ in range(num_epochs):
        model.train()
        for x, y, _ in retain_loader:
            x, y = x.to(device), y.to(device)
            _, logits = model(x, y)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_neggrad(model, forget_loader, retain_loader, cfg, num_epochs=5):
    """Baseline: NegGrad / NegGrad+ (gradient ascent trên forget set)."""
    device = cfg.device; model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=getattr(cfg, 'neggrad_lr', 1e-5), momentum=0.9)
    ce  = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)
    for _ in range(num_epochs):
        model.train()
        for x_r, y_r, _ in retain_loader:
            x_r, y_r = x_r.to(device), y_r.to(device)
            _, logits_r = model(x_r, y_r); loss_r = ce(logits_r, y_r)
            try: x_f, y_f, _ = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                x_f, y_f, _ = next(forget_iter)
            x_f, y_f = x_f.to(device), y_f.to(device)
            _, logits_f = model(x_f, y_f)
            loss = loss_r - ce(logits_f, y_f)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model