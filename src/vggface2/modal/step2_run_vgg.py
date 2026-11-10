"""
BƯỚC 2/3 — Chạy thí nghiệm VGGFace2 trên GPU của Modal.

  Chạy thử (đo tốc độ, ~10–20 phút):   modal run step2_run_vgg.py --quick
  Chạy đầy đủ (nên dùng --detach):      modal run --detach step2_run_vgg.py --seed 42

Đọc dữ liệu từ volume "lpeu-vgg-cache" (bước 1), ghi kết quả vào
lpeu-vgg-cache/runs/vgg_seed<seed>[_quick]/ : results.json, signals.npz, các *.pt
Nếu run bị ngắt giữa chừng, chạy lại cùng seed sẽ tự nạp original_model.pt / retrain_model.pt đã lưu.
"""
from pathlib import Path
import modal

HERE = Path(__file__).parent
app = modal.App("lpeu-vgg-run")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "scikit-learn", "pandas", "pillow", "numpy")
    .add_local_file(HERE / "lpeu_core.py", "/root/lpeu_core.py")
    .add_local_file(HERE / "vgg_pipeline.py", "/root/vgg_pipeline.py")
)
vol = modal.Volume.from_name("lpeu-vgg-cache")


@app.function(image=image, gpu="L40S", cpu=16, memory=48 * 1024,
              timeout=5 * 60 * 60, volumes={"/data": vol})   # tối đa 5 giờ ≈ $15 trường hợp xấu nhất
def run(seed: int = 42, quick: bool = False, tag: str = ""):
    import sys
    sys.path.insert(0, "/root")
    import vgg_pipeline as V
    try:
        return V.pipeline(seed=seed, quick=quick, data_dir="/data/vgg", out_dir="/data/runs", tag=tag)
    finally:
        vol.commit()


@app.local_entrypoint()
def main(seed: int = 42, quick: bool = False, tag: str = ""):
    print(run.remote(seed=seed, quick=quick, tag=tag))
