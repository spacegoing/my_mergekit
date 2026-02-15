#!/usr/bin/env python3
"""
Batch SVD merge runner — launches parallel merges across GPUs.

Each job merges one RL checkpoint with the base model using svd_merge.py.
Jobs are assigned to GPUs round-robin. Up to NUM_GPUS jobs run simultaneously;
when a GPU finishes, the next queued job starts on that GPU.

Usage:
  python single_merge/run_svd_batch.py

Edit JOBS below to specify your checkpoints.
"""

import os
import subprocess
import sys
import time

# ===================== EDIT THESE =====================
BASE_DIR = "/root/myCodeLab/host/downloads/models/40Bv6/dpo-0210-0208-v2-dpoaddid-965/965"
OUT_ROOT = "./merged"   # outputs go to {OUT_ROOT}/{name}_{combo}/
NUM_GPUS = 8
COMBOS = ["c", "b", "a"]  # combos to run for each checkpoint

# (rl_checkpoint_path, short_name)
CHECKPOINTS = [
    ("/root/myCodeLab/host/verl/ckpts/single_domain/sd_c351_facpo_nemogym_math_d0.5-tp1.5-tn2.0-ent0-bdm1-ppoch2-1cb2094f_20260213_184907/global_step_80/actor/huggingface",
     "c351_svs8"),
    # ("/root/myCodeLab/host/verl/ckpts/single_domain/sd_c411_facpo_nemogym_math_d1.0-tp1.5-tn2.0-ent0-bdm1-ppoch2-575c58dc_20260214_211417/global_step_60/actor/huggingface",
    #  "c411_svs6"),
    # ("/root/myCodeLab/host/verl/ckpts/single_domain/sd_c411_facpo_nemogym_math_d1.0-tp1.5-tn2.0-ent0-bdm1-ppoch2-575c58dc_20260214_211417/global_step_80/actor/huggingface",
    # "c411_svs8"),
]
# ======================================================


def main():
    if not CHECKPOINTS:
        print("No checkpoints defined. Edit CHECKPOINTS list in run_svd_batch.py.")
        sys.exit(1)
    if not COMBOS:
        print("No combos defined. Edit COMBOS list in run_svd_batch.py.")
        sys.exit(1)

    # Build job list: cross-product of checkpoints × combos
    # Each job is (rl_dir, name, combo)
    jobs = []
    for rl_dir, name in CHECKPOINTS:
        for combo in COMBOS:
            jobs.append((rl_dir, name, combo))

    script = os.path.join(os.path.dirname(__file__), "svd_merge.py")
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"Batch SVD merge: {len(CHECKPOINTS)} ckpts × {len(COMBOS)} combos = {len(jobs)} jobs, {NUM_GPUS} GPUs")
    print(f"  base:    {BASE_DIR}")
    print(f"  combos:  {COMBOS}")
    print(f"  output:  {OUT_ROOT}")
    print()

    # Track running processes: gpu_id -> (job_label, Popen, log_file_handle)
    gpu_slots = {}
    job_queue = list(enumerate(jobs))
    completed = []
    failed = []

    def launch_job(gpu_id, idx, rl_dir, name, combo):
        job_label = f"{name}_svd{combo}"
        out_dir = os.path.join(OUT_ROOT, job_label)
        log_file = os.path.join(OUT_ROOT, f"{job_label}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, script,
            "--base_dir", BASE_DIR,
            "--rl_dir", rl_dir,
            "--out_dir", out_dir,
            "--combo", combo,
            "--device", "cuda",
        ]
        log_f = open(log_file, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        print(f"  [{idx+1}/{len(jobs)}] GPU {gpu_id}: {job_label}  (pid={p.pid})")
        gpu_slots[gpu_id] = (job_label, p, log_f)

    def check_slots():
        """Check for completed processes and free their GPU slots."""
        freed = []
        for gpu_id, (label, proc, log_f) in list(gpu_slots.items()):
            ret = proc.poll()
            if ret is not None:
                log_f.close()
                if ret == 0:
                    print(f"  GPU {gpu_id}: {label} done.")
                    completed.append(label)
                else:
                    print(f"  GPU {gpu_id}: {label} FAILED (exit={ret})")
                    failed.append(label)
                freed.append(gpu_id)
        for gpu_id in freed:
            del gpu_slots[gpu_id]
        return freed

    # Initial launch: fill all GPU slots
    print("Launching jobs:")
    while job_queue and len(gpu_slots) < NUM_GPUS:
        idx, (rl_dir, name, combo) = job_queue.pop(0)
        gpu_id = len(gpu_slots)
        launch_job(gpu_id, idx, rl_dir, name, combo)

    # Wait loop: as GPUs free up, launch remaining jobs
    while gpu_slots:
        time.sleep(5)
        freed = check_slots()
        for gpu_id in freed:
            if job_queue:
                idx, (rl_dir, name, combo) = job_queue.pop(0)
                launch_job(gpu_id, idx, rl_dir, name, combo)

    # Summary
    print()
    print("=" * 50)
    print(f"Completed: {len(completed)}/{len(jobs)}")
    for label in completed:
        print(f"  {label}")
    if failed:
        print(f"Failed: {len(failed)}/{len(jobs)}")
        for label in failed:
            print(f"  {label}  (check {OUT_ROOT}/{label}.log)")


if __name__ == "__main__":
    main()
