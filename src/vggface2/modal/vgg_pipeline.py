"""
VGGFace2 pipeline cho bài FJCAI — cùng giao thức "Study B" như MUFAC resubmit, áp dụng
cho thiết lập identity = class:

  * dữ liệu đọc từ 1 mảng numpy trong RAM (images_u8.npy) → không giải nén JPEG mỗi epoch
  * 10% identity giữ ngoài train (unseen), chia đôi: REF (chỉ cho L_div) / TEST (chỉ cho đánh giá)
  * 100 identity bị quên; Original, Retrain (chỉ retain), FineTune, NegGrad,
    LPEU (per-person, L_div) và LPEU không L_div; chọn checkpoint trên retain-VAL
  * lưu tín hiệu TỪNG ẢNH (confidence, loss) để tính FS trên toàn bộ dữ liệu + bootstrap theo identity

Hàm chính: pipeline(seed, quick, data_dir, out_dir, device)
"""
import os, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import lpeu_core as L   # các cell của notebook (Config, FaceModel, LPEU loop, ...)


# ─────────────────────────── dữ liệu ───────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class NpyFaceDataset(Dataset):
    """Ảnh uint8 [N,H,W,3] trong RAM; trả (x, y). y = class đã remap (-1 với unseen)."""
    def __init__(self, X, y, keys, train):
        self.X, self.y, self.train = X, y, train
        self.households = keys          # LPEU core dùng .households làm khoá group
        self.person_ids = ["p"] * len(y)
        self.samples = [(None, int(v)) for v in y]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.X[i])).permute(2, 0, 1).float().div_(255.0)
        if self.train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            b = 1.0 + (random.random() - 0.5) * 0.4
            c = 1.0 + (random.random() - 0.5) * 0.4
            m = x.mean()
            x = ((x - m) * c + m) * b
            x = x.clamp_(0, 1)
        x = (x - _MEAN) / _STD
        return x, int(self.y[i])


def split_vgg(labels_raw, ids, seed, n_forget=100, unseen_ratio=0.10):
    rng = random.Random(seed)
    n_id = len(ids)
    order = list(range(n_id)); rng.shuffle(order)
    n_unseen = int(round(n_id * unseen_ratio))
    unseen_ids = sorted(order[:n_unseen]); train_ids = sorted(order[n_unseen:])
    rng2 = random.Random(seed + 1000)
    u = unseen_ids[:]; rng2.shuffle(u)
    unseen_ref_ids, unseen_test_ids = sorted(u[: len(u) // 2]), sorted(u[len(u) // 2:])
    forget_ids = sorted(rng.sample(train_ids, n_forget))
    remap = {old: new for new, old in enumerate(train_ids)}      # class index cho identity huấn luyện

    by_id = {}
    for i, c in enumerate(labels_raw):
        by_id.setdefault(int(c), []).append(i)
    S = {k: [] for k in ["train", "val", "test", "unseen_ref", "unseen_test"]}
    for c in train_ids:
        idx = by_id[c][:]; rng.shuffle(idx); n = len(idx)
        n_tr = max(1, int(n * 0.7)); n_va = max(1, int(n * 0.1))
        S["train"] += idx[:n_tr]; S["val"] += idx[n_tr:n_tr + n_va]; S["test"] += idx[n_tr + n_va:]
    for c in unseen_ref_ids:
        S["unseen_ref"] += by_id[c]
    for c in unseen_test_ids:
        S["unseen_test"] += by_id[c]
    fset = set(forget_ids)
    is_f = lambda i: int(labels_raw[i]) in fset
    S["forget"] = [i for i in S["train"] if is_f(i)]
    S["retain_train"] = [i for i in S["train"] if not is_f(i)]
    S["retain_val"] = [i for i in S["val"] if not is_f(i)]
    S["retain_test"] = [i for i in S["test"] if not is_f(i)]
    y_class = np.array([remap.get(int(c), -1) for c in labels_raw], dtype=np.int64)
    keys = [ids[int(c)] for c in labels_raw]
    return S, y_class, keys, [remap[c] for c in forget_ids], [ids[c] for c in forget_ids], len(train_ids)


# ─────────────────────────── tín hiệu MIA từng ảnh ───────────────────────────
@torch.no_grad()
def per_sample_signals(model, loader, device):
    """max-softmax confidence (không cần nhãn) + CE loss (nan nếu không có nhãn)."""
    model.eval(); conf, loss = [], []
    for x, y, _ in loader:
        _, logits = model(x.to(device), None)
        p = F.softmax(logits.float(), dim=1)
        conf.append(p.max(1).values.cpu())
        yy = y.to(device)
        ok = yy >= 0
        l = torch.full((len(yy),), float("nan"), device=device)
        if ok.any():
            l[ok] = F.cross_entropy(logits.float()[ok], yy[ok], reduction="none")
        loss.append(l.cpu())
    return torch.cat(conf).numpy(), torch.cat(loss).numpy()


def _fs_full(member, nonmember):
    """FS = |0.5 - acc| và |0.5 - AUC| của LR 1-D, cân bằng 2 lớp, trên TOÀN BỘ mẫu."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    k = min(len(member), len(nonmember))
    a = member[np.linspace(0, len(member) - 1, k).astype(int)]
    b = nonmember[np.linspace(0, len(nonmember) - 1, k).astype(int)]
    X = np.concatenate([a, b]).reshape(-1, 1); y = np.r_[np.ones(k), np.zeros(k)]
    clf = LogisticRegression(max_iter=500).fit(X, y)
    return abs(0.5 - clf.score(X, y)), abs(0.5 - roc_auc_score(y, clf.predict_proba(X)[:, 1]))



@torch.no_grad()
def fast_reid(model, base, forget_key_idx, retain_key_idx, batch_size, num_workers, device):
    """Re-ID retrieval giống L.compute_reidentification_accuracy (gallery = 1 prototype/identity,
    query = từng ảnh forget, prototype của chính identity đó tính leave-one-out), nhưng tính
    embedding MỘT lần cho mọi ảnh rồi xếp hạng bằng phép nhân ma trận (bản gốc tạo 1 DataLoader
    cho mỗi identity → ~8 phút/mô hình chỉ để khởi tạo worker)."""
    model.eval()
    all_idx = [i for v in retain_key_idx.values() for i in v] + [i for v in forget_key_idx.values() for i in v]
    loader = L.make_loader(L.IndexedSubset(base, all_idx), batch_size, False, num_workers)
    Z = torch.cat([model(x.to(device), None)[0].float().cpu() for x, _, _ in loader])
    pos = {idx: r for r, idx in enumerate(all_idx)}
    keys, protos, sums = [], [], {}
    for k, v in list(retain_key_idx.items()) + list(forget_key_idx.items()):
        e = Z[[pos[i] for i in v]]
        sums[k] = e
        keys.append(k); protos.append(F.normalize(e.mean(0, keepdim=True), dim=1))
    G = torch.cat(protos)                                   # [N, D]
    row = {k: r for r, k in enumerate(keys)}
    ranks, skipped = [], 0
    for k in forget_key_idx:
        e = sums[k]; n = e.size(0)
        if n < 2:
            skipped += 1; continue
        q = F.normalize(e, dim=1)                             # [n, D]
        loo = F.normalize(e.sum(0, keepdim=True) - e, dim=1)  # [n, D] prototype bỏ chính ảnh đó
        self_sim = (q * loo).sum(1)                           # [n]
        sims = q @ G.T                                        # [n, N]
        sims[:, row[k]] = -float("inf")                       # loại hàng của chính identity
        ranks.append((sims > self_sim[:, None]).sum(1) + 1)
    r = torch.cat(ranks).float()
    N = G.size(0)
    return dict(reid_accuracy=float((r == 1).float().mean()), reid_top5_accuracy=float((r <= 5).float().mean()),
                reid_top10_accuracy=float((r <= 10).float().mean()), reid_mean_rank=float(r.mean()),
                reid_median_rank=float(r.median()), reid_chance_mean_rank=(N + 1) / 2.0,
                reid_n_queries=int(len(r)), reid_gallery_size=N,
                reid_n_persons_evaluated=len(forget_key_idx) - skipped)

# ─────────────────────────── pipeline ───────────────────────────
def pipeline(seed=42, quick=False, data_dir="/data/vgg", out_dir="/data/runs", device=None,
             run_retrain=True, run_ablation_nodiv=True, num_workers=8, tag=""):
    t0 = time.time()
    tick = lambda m: print(f"\n[{(time.time()-t0)/60:6.1f} min] {m}", flush=True)
    cfg = L.cfg
    cfg.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.seed = seed
    base_dir = os.path.join(out_dir, f"vgg_seed{seed}" + ("_quick" if quick else ""))
    cfg.output_dir = base_dir + tag   # tag != "" → kết quả mới, dùng lại Original/Retrain của base_dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.num_workers = num_workers
    # ── cấu hình riêng VGGFace2 (giống notebook VGGFace2 trước đây) ──
    cfg.batch_size = 64; cfg.checkpoint_retain_gate = 0.99; cfg.forget_manual_lr = 1e-3   # v2: về mặc định như MUFAC (2e-4 làm nhánh forget đứng yên)
    cfg.stage1_epochs, cfg.stage2_epochs, cfg.stage3_epochs = 1, 14, 5
    cfg.w_kl_uniform = 3.0; cfg.w_div = 20.0
    cfg.repair_epochs = 30; cfg.repair_layer4_lr_mult = 0.2; cfg.repair_forget_guard_weight = 0.3
    cfg.k_per_person, cfg.k_anchor_max, cfg.k_proj = 3, 150, 30
    cfg.finetune_lr, cfg.neggrad_lr = 1e-3, 1e-4   # v2: baseline đủ mạnh (NegGrad 1e-5 không quên gì)
    if quick:   # chạy thử: đo tốc độ, kiểm tra code
        cfg.original_epochs = 2; cfg.warmup_epochs = 1; cfg.early_stop_patience = 2
        cfg.stage1_epochs, cfg.stage2_epochs, cfg.stage3_epochs = 1, 1, 1
        cfg.repair_epochs = 2; cfg.migrate_max_epochs = 1
    cfg.unlearn_epochs = cfg.stage1_epochs + cfg.stage2_epochs + cfg.stage3_epochs
    L.set_seed(seed)
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    tick("[1] Load data")
    X = np.load(os.path.join(data_dir, "images_u8.npy"))
    labels_raw = np.load(os.path.join(data_dir, "labels.npy"))
    ids = json.load(open(os.path.join(data_dir, "meta.json")))["identities"]
    S, y_class, keys, forget_cls, forget_keys, n_cls = split_vgg(labels_raw, ids, seed)
    print(f"  images={len(labels_raw)} identities={len(ids)} trainable={n_cls} forget=100 "
          f"| train={len(S['train'])} forget_imgs={len(S['forget'])} retain_test={len(S['retain_test'])} "
          f"| unseen_ref={len(S['unseen_ref'])} unseen_test={len(S['unseen_test'])}")
    train_base = NpyFaceDataset(X, y_class, keys, train=True)
    eval_base = NpyFaceDataset(X, y_class, keys, train=False)
    LD = lambda base, idx, sh: L.make_loader(L.IndexedSubset(base, idx), cfg.batch_size, sh, cfg.num_workers)
    train_loader = LD(train_base, S["train"], True); val_loader = LD(eval_base, S["val"], False)
    retain_train_loader = LD(train_base, S["retain_train"], True)
    retain_val_loader = LD(eval_base, S["retain_val"], False)
    retain_test_loader = LD(eval_base, S["retain_test"], False)
    forget_train_loader = LD(train_base, S["forget"], True)
    forget_eval_loader = LD(eval_base, S["forget"], False)
    unseen_test_loader = LD(eval_base, S["unseen_test"], False)
    ref = S["unseen_ref"][:]; random.Random(seed).shuffle(ref); ref = ref[:2000]
    unseen_ref_imgs = {-1: torch.stack([eval_base[i][0] for i in ref])}   # L_div lấy mẫu đều (không khớp lớp)

    def new_model():
        return L.FaceModel(num_classes=n_cls, embedding_dim=cfg.embedding_dim,
                           pretrained=cfg.pretrained_backbone, s=cfg.arcface_s, m=cfg.arcface_m)

    def train_or_load(name, tl, vl):
        ck = os.path.join(cfg.output_dir, f"{name}.pt")
        if not os.path.exists(ck) and os.path.exists(os.path.join(base_dir, f"{name}.pt")):
            ck = os.path.join(base_dir, f"{name}.pt")   # dùng lại model đã train → cùng Original, tiết kiệm GPU
        L.set_seed(seed); m = new_model()
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=cfg.device)); m.to(cfg.device)
            print(f"  → loaded {ck}")
        else:
            m = L.train_original_model(m, tl, vl, cfg); torch.save(m.state_dict(), ck)
        return m

    tick("[2] Train ORIGINAL"); original = train_or_load("original_model", train_loader, val_loader)
    retrain = None
    if run_retrain:
        tick("[2b] Train RETRAIN (retain only)")
        retrain = train_or_load("retrain_model", retain_train_loader, retain_val_loader)

    tick("[3] Per-identity prototypes & K-NN")
    all_protos = L.compute_class_prototypes(original, LD(eval_base, S["train"], False), n_cls, cfg.device)
    old_forget_proto = L.compute_forget_prototype(original, forget_eval_loader, cfg.device)
    person_protos = L.compute_group_prototypes(original, LD(eval_base, S["train"], False), keys, cfg.device)
    person_lookup, forget_order = L.build_person_lookup(keys, forget_keys)
    person_protos_old = L.stack_person_protos(person_protos, forget_order)
    anchor_ids, proj_ids = L.find_knn_retain_persons(person_protos, forget_keys,
                                                     cfg.k_per_person, cfg.k_anchor_max, cfg.k_proj)
    local_protos = torch.stack([person_protos[p] for p in proj_ids])
    anchor_loader = L.build_local_anchor_loader_grouped(eval_base, S["retain_train"], keys, anchor_ids,
                                                        cfg.batch_size, cfg.num_workers)
    aset = set(anchor_ids)
    local_loader = LD(eval_base, [i for i in S["retain_train"] if keys[i] in aset], False)

    tick("[4] EWC"); theta_star = L.snapshot_neck_params(original)
    fisher = L.compute_ewc_fisher(original, retain_train_loader, cfg.device, n_batches=30)

    tick("[5] Baselines")
    L.set_seed(seed); ft = L.run_finetune(L.clone_model(original), retain_train_loader, cfg)
    L.set_seed(seed); ng = L.run_neggrad(L.clone_model(original), forget_train_loader, retain_train_loader, cfg)
    ref_val = max(L.evaluate_accuracy(original, retain_val_loader, cfg.device), 1e-6)
    best_base_val = max(L.evaluate_accuracy(ft, retain_val_loader, cfg.device),
                        L.evaluate_accuracy(ng, retain_val_loader, cfg.device))

    def lpeu(tag, use_div):
        tick(f"[6] LPEU {tag} (L_div={'ON' if use_div else 'OFF'}, per-person)")
        L.set_seed(seed)
        m = L.run_lpeu_v7_unlearning(
            model=L.clone_model(original), old_model=L.clone_model(original),
            forget_loader=forget_train_loader, retain_loader=retain_train_loader,
            retain_test_loader=retain_val_loader, anchor_loader=anchor_loader,
            forget_ids=forget_cls, num_classes=n_cls, cfg=cfg, retain_acc_ref=ref_val,
            fisher=fisher, theta_star=theta_star, retain_prototypes=all_protos,
            local_retain_prototypes=local_protos,
            unseen_by_class_images=(unseen_ref_imgs if use_div else None),
            person_lookup=person_lookup, person_protos_old=person_protos_old)
        if cfg.use_post_repair and L.evaluate_accuracy(m, retain_val_loader, cfg.device) < ref_val:
            m = L.run_retain_repair(
                model=m, old_model=L.clone_model(original), retain_loader=retain_train_loader,
                retain_test_loader=retain_val_loader, forget_ids=forget_cls, num_classes=n_cls, cfg=cfg,
                retain_prototypes=all_protos, retain_val_loader=retain_val_loader,
                forget_loader=forget_train_loader, forget_proto=old_forget_proto,
                retain_acc_ref=ref_val, beat_acc=best_base_val)
        torch.save(m.state_dict(), os.path.join(cfg.output_dir, f"lpeu_{tag}.pt"))
        return m

    models = {"Original": original}
    if retrain is not None: models["Retrain"] = retrain
    models.update(FineTune=ft, NegGrad=ng, LPEU=lpeu("full", True))
    if run_ablation_nodiv: models["LPEU-noDiv"] = lpeu("nodiv", False)

    tick("[7] Evaluation")
    f_key_idx, r_key_idx = {}, {}
    for i in S["forget"]: f_key_idx.setdefault(keys[i], []).append(i)
    for i in S["retain_train"]: r_key_idx.setdefault(keys[i], []).append(i)
    res, dump = {}, {"forget_key": np.array([keys[i] for i in S["forget"]]),
                     "retain_test_key": np.array([keys[i] for i in S["retain_test"]]),
                     "unseen_test_key": np.array([keys[i] for i in S["unseen_test"]])}
    for name, m in models.items():
        fc, fl = per_sample_signals(m, forget_eval_loader, cfg.device)
        rc, rl = per_sample_signals(m, retain_test_loader, cfg.device)
        uc, _ = per_sample_signals(m, unseen_test_loader, cfg.device)
        dump.update({f"{name}__forget_conf": fc, f"{name}__forget_loss": fl, f"{name}__retain_conf": rc,
                     f"{name}__retain_loss": rl, f"{name}__unseen_conf": uc})
        fs_u_acc, fs_u_auc = _fs_full(fc, uc)            # quên identity: forget vs người chưa từng thấy
        fs_r_acc, fs_r_auc = _fs_full(-np.nan_to_num(fl), -np.nan_to_num(rl))   # forget vs retain-test (loss)
        r = dict(retain_acc=L.evaluate_accuracy(m, retain_test_loader, cfg.device),
                 forget_acc=L.evaluate_accuracy(m, forget_eval_loader, cfg.device),
                 FS_unseen_conf_acc=fs_u_acc, FS_unseen_conf_auc=fs_u_auc,
                 FS_retain_loss_acc=fs_r_acc, FS_retain_loss_auc=fs_r_auc,
                 forget_intra_cos=L.intra_cluster_cos_sim(m, forget_eval_loader, cfg.device))
        if name not in ("Original", "Retrain"):
            r["neighbor_shift"] = L.neighbor_shift(m, original, local_loader, cfg.device)
            r["global_shift"] = L.neighbor_shift(m, original, retain_test_loader, cfg.device)
            r["forget_proto_sim"] = L.compute_person_proto_sim(m, forget_eval_loader, person_lookup,
                                                               person_protos_old, cfg.device)
        r.update(fast_reid(m, eval_base, f_key_idx, r_key_idx, cfg.batch_size, cfg.num_workers, cfg.device))
        res[name] = r
        print(f"  {name:11s} ret={r['retain_acc']:.4f} fgt={r['forget_acc']:.4f} "
              f"FS_unseen(acc/auc)={fs_u_acc:.3f}/{fs_u_auc:.3f} FS_retain(acc/auc)={fs_r_acc:.3f}/{fs_r_auc:.3f} "
              f"intra={r['forget_intra_cos']:.3f} reid_top1={r['reid_accuracy']:.3f} "
              f"rank={r['reid_mean_rank']:.1f} nbr={r.get('neighbor_shift', float('nan')):.4f}", flush=True)

    out = dict(seed=seed, quick=quick, n_identities=len(ids), n_trainable=n_cls,
               sizes={k: len(v) for k, v in S.items()}, forget_identities=forget_keys,
               anchor_ids=anchor_ids, results=res, runtime_min=(time.time() - t0) / 60,
               config={k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool))})
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    np.savez_compressed(os.path.join(cfg.output_dir, "signals.npz"), **dump)
    tick(f"DONE → {cfg.output_dir}")
    return {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                if kk in ("retain_acc", "forget_acc", "FS_unseen_conf_acc", "FS_retain_loss_acc",
                          "forget_intra_cos", "reid_accuracy", "neighbor_shift")} for k, v in res.items()}
