"""
BƯỚC 1 — Chuẩn bị dữ liệu VGGFace2 trên Modal từ Volume CÓ SẴN (chỉ dùng CPU).

Nguồn : Volume "lpeu-vggface2" (đã có, ~500k ảnh, đường dẫn .../vggface2_data/train/nXXXXXX/)
        → KHÔNG tải lại từ Kaggle, KHÔNG cần Kaggle secret.
        Volume này là loại v1 và đã chạm giới hạn ~500k file → chỉ ĐỌC, không ghi vào.
Đích  : Volume MỚI "lpeu-vgg-cache" (tự tạo) chứa:
          vgg/images_u8.npy  [N,128,128,3] uint8
          vgg/labels.npy     [N] int32
          vgg/meta.json
        → khi train chỉ cần np.load — không giải nén JPEG mỗi epoch nữa.

Chọn dữ liệu: tối đa 1.500 identity có >= 20 ảnh (dùng hết nếu volume có ít hơn) (seed 42), mỗi identity tối đa 40 ảnh,
resize 128x128 — cùng quy mô với run VGGFace2 trong bài.
Identity cuối cùng theo thứ tự tên bị bỏ ra (có thể bị cắt dở khi giải nén chạm giới hạn file).

Chạy:   modal run step1_prepare_data.py
"""
import modal

N_IDENTITIES = 1500
MIN_IMAGES = 20
MAX_IMAGES_PER_ID = 40
IMG = 128
SEED = 42
OUT = "/data/vgg"          # thư mục kết quả trong volume lpeu-vgg-cache

app = modal.App("lpeu-vgg-prepare")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pillow", "numpy")
src_vol = modal.Volume.from_name("lpeu-vggface2")                       # có sẵn, chỉ đọc
dst_vol = modal.Volume.from_name("lpeu-vgg-cache", create_if_missing=True)


def _resize_identity(paths):
    """Chạy trong process con: đọc + resize ảnh của 1 identity."""
    import numpy as np
    from PIL import Image
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((IMG, IMG), Image.BILINEAR)
                out.append(np.asarray(im, dtype=np.uint8))
        except Exception as e:
            print(f"  ! skip {p}: {e}")
    return out


def _find_train_dir(root):
    """Tìm thư mục 'train' chứa các folder nXXXXXX bên trong volume."""
    import os, re
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == "train" and any(re.fullmatch(r"n\d+", d) for d in dirnames[:50]):
            return dirpath
        if dirpath.count(os.sep) - root.count(os.sep) >= 3:   # không đi quá sâu
            dirnames[:] = []
    raise FileNotFoundError("Không tìm thấy thư mục train/nXXXXXX trong volume lpeu-vggface2")


@app.function(
    image=image,
    cpu=16,
    memory=24 * 1024,
    timeout=3 * 60 * 60,
    volumes={"/src": src_vol, "/data": dst_vol},
)
def prepare():
    import os, re, json, random, time
    from multiprocessing import Pool
    import numpy as np

    t0 = time.time()
    train = _find_train_dir("/src")
    print(f"[1/4] Thư mục nguồn: {train}", flush=True)

    ids = sorted(d for d in os.listdir(train) if re.fullmatch(r"n\d+", d))
    print(f"      {len(ids)} identity; bỏ identity cuối {ids[-1]} (có thể bị cắt dở)", flush=True)
    ids = ids[:-1]

    print("[2/4] Đếm ảnh mỗi identity ...", flush=True)
    files = {}
    for pid in ids:
        fs = [f for f in os.listdir(os.path.join(train, pid))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(fs) >= MIN_IMAGES:
            files[pid] = sorted(fs)
    eligible = sorted(files)
    print(f"      {len(eligible)} identity có >= {MIN_IMAGES} ảnh ({(time.time()-t0)/60:.1f} phút)")
    n_sel = min(N_IDENTITIES, len(eligible))
    if n_sel < N_IDENTITIES:
        print(f"      ⚠ Volume chỉ có {len(eligible)} identity đủ điều kiện (< {N_IDENTITIES}) "
              f"→ dùng toàn bộ {n_sel} identity", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(eligible)
    selected = sorted(eligible[:n_sel])
    jobs = []
    for pid in selected:
        fs = list(files[pid])
        rng.shuffle(fs)
        jobs.append([os.path.join(train, pid, f) for f in fs[:MAX_IMAGES_PER_ID]])

    print(f"[3/4] Resize {sum(map(len, jobs))} ảnh của {len(selected)} identity "
          f"bằng {os.cpu_count()} CPU ...", flush=True)
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(_resize_identity, jobs, chunksize=8)

    images, labels = [], []
    for cls, arrs in enumerate(results):
        images.extend(arrs)
        labels.extend([cls] * len(arrs))
    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int32)

    print("[4/4] Ghi vào volume lpeu-vgg-cache ...", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/images_u8.npy", X)
    np.save(f"{OUT}/labels.npy", y)
    counts = np.bincount(y)
    meta = dict(identities=selected, n_images=int(len(y)), n_identities=len(selected),
                img_size=IMG, min_images=MIN_IMAGES, max_images_per_id=MAX_IMAGES_PER_ID,
                seed=SEED, images_per_id_min=int(counts.min()),
                images_per_id_max=int(counts.max()), source_train_dir=train)
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f)
    dst_vol.commit()
    print(f"\nXONG trong {(time.time()-t0)/60:.1f} phút: {X.shape}, "
          f"{counts.min()}-{counts.max()} ảnh/identity, {X.nbytes/1e9:.2f} GB")
    return {k: v for k, v in meta.items() if k != "identities"}


@app.local_entrypoint()
def main():
    print(prepare.remote())
