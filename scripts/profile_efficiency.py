"""
Lightweight profiling script for N-HiTS, TFT, and Chronos-2.

Measures:
  1. Training duration (N-HiTS, TFT) / model-load time (Chronos-2)
  2. Inference latency per 7-day window batch
  3. Peak RAM / VRAM during inference

Outputs a clean markdown table.

Usage:
    python scripts/profile_efficiency.py
"""

import time
import yaml

import psutil
import torch

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.data_prep import to_chronos_context_df
from agri_agent.forecasting.evaluate import (
    historical_slice_to_future_df,
    temporal_train_test_split,
)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
HORIZON_DAYS = 7
N_WARMUP = 2      # warm-up inference runs before timing
N_TIMED  = 5      # timed inference runs


def load_field_config(path="configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_proc = psutil.Process()


def fmt_secs(seconds: float) -> str:
    """Format seconds as e.g. '127.3 s' or '2 m 7 s'."""
    if seconds >= 120:
        m, s = divmod(seconds, 60)
        return f"{int(m)} m {s:.1f} s"
    return f"{seconds:.1f} s"


def sample_rss_mb() -> float:
    """Return current RSS in MB."""
    return _proc.memory_info().rss / 1024 / 1024


def reset_vram_counter():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0



def profile_nhits(train_df, test_df, field_id):
    from agri_agent.forecasting import nhits_model, nf_data_prep

    nf_train_df = nf_data_prep.to_neuralforecast_df(train_df)

    # --- Training ---
    t0 = time.perf_counter()
    nf = nhits_model.train_nhits(nf_train_df, HORIZON_DAYS)
    train_time = time.perf_counter() - t0

    # --- Inference (warm-up + timed) ---
    futr_df = nf_data_prep.historical_slice_to_futr_df(test_df, field_id)

    for _ in range(N_WARMUP):
        _ = nhits_model.predict_nhits(nf, futr_df)

    reset_vram_counter()
    rss_steady = sample_rss_mb()
    infer_t0 = time.perf_counter()
    for _ in range(N_TIMED):
        _ = nhits_model.predict_nhits(nf, futr_df)
    infer_time = (time.perf_counter() - infer_t0) / N_TIMED
    peak_vram = get_peak_vram_mb()

    return train_time, infer_time, rss_steady, peak_vram


def profile_tft(train_df, test_df, field_id):
    from agri_agent.forecasting import nf_data_prep, tft_model

    nf_train_df = nf_data_prep.to_neuralforecast_df(train_df)

    # --- Training ---
    t0 = time.perf_counter()
    nf = tft_model.train_tft(nf_train_df, HORIZON_DAYS)
    train_time = time.perf_counter() - t0

    # --- Inference (warm-up + timed) ---
    futr_df = nf_data_prep.historical_slice_to_futr_df(test_df, field_id)

    for _ in range(N_WARMUP):
        _ = tft_model.predict_tft(nf, futr_df)

    reset_vram_counter()
    rss_steady = sample_rss_mb()
    infer_t0 = time.perf_counter()
    for _ in range(N_TIMED):
        _ = tft_model.predict_tft(nf, futr_df)
    infer_time = (time.perf_counter() - infer_t0) / N_TIMED
    peak_vram = get_peak_vram_mb()

    return train_time, infer_time, rss_steady, peak_vram


def profile_chronos(train_df, test_df, field_id):
    from chronos import Chronos2Pipeline

    from agri_agent.forecasting.chronos_model import (
        MODEL_ID,
        _fill_covariate_gaps,
        _select_device,
    )

    context_df = to_chronos_context_df(train_df)
    futr_df = historical_slice_to_future_df(test_df, field_id)
    context_df = _fill_covariate_gaps(
        context_df, ["precipitation_sum", "et0_fao_evapotranspiration"]
    )
    futr_df = _fill_covariate_gaps(
        futr_df, ["precipitation_sum", "et0_fao_evapotranspiration"]
    )

    device = _select_device()

    # --- Model-load time (Chronos-2 is zero-shot, no training) ---
    t0 = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)
    load_time = time.perf_counter() - t0

    # --- Inference (warm-up + timed) ---
    for _ in range(N_WARMUP):
        _ = pipeline.predict_df(
            context_df, future_df=futr_df, prediction_length=HORIZON_DAYS,
            quantile_levels=[0.5], id_column="id", timestamp_column="timestamp",
            target="target",
        )

    reset_vram_counter()
    rss_steady = sample_rss_mb()
    infer_t0 = time.perf_counter()
    for _ in range(N_TIMED):
        _ = pipeline.predict_df(
            context_df, future_df=futr_df, prediction_length=HORIZON_DAYS,
            quantile_levels=[0.5], id_column="id", timestamp_column="timestamp",
            target="target",
        )
    infer_time = (time.perf_counter() - infer_t0) / N_TIMED
    peak_vram = get_peak_vram_mb()

    return load_time, infer_time, rss_steady, peak_vram


def main():
    field = load_field_config()
    field_id = field["field_id"]

    print("Loading fused dataset ...")
    fused_df = load_fused_dataset(FUSED_PARQUET)
    print(f"  {len(fused_df)} rows, {fused_df['date'].min().date()} to "
          f"{fused_df['date'].max().date()}\n")

    train_df, test_df = temporal_train_test_split(fused_df, HORIZON_DAYS)
    print(f"  Train: {len(train_df)} days, Test: {len(test_df)} days\n")

    rss_baseline = sample_rss_mb()
    print(f"  Baseline RSS (Python + data loaded): {rss_baseline:.0f} MB\n")

    results = {}

    # ---- N-HiTS ----
    print("Profiling N-HiTS (no pooling + target-only) ...")
    t_train, t_infer, peak_ram, peak_vram = profile_nhits(
        train_df, test_df, field_id
    )
    ram_delta = peak_ram - rss_baseline
    results["N-HiTS"] = {
        "train_time": t_train,
        "infer_time": t_infer,
        "ram_total_mb": peak_ram,
        "ram_delta_mb": ram_delta,
        "peak_vram_mb": peak_vram,
    }
    print(f"  Train: {fmt_secs(t_train)}, Infer: {t_infer*1000:.1f} ms/win, "
          f"RAM: {peak_ram:.0f} MB  (+{ram_delta:.0f} MB vs baseline), "
          f"VRAM: {peak_vram:.0f} MB\n")

    # ---- TFT ----
    print("Profiling TFT ...")
    t_train, t_infer, peak_ram, peak_vram = profile_tft(
        train_df, test_df, field_id
    )
    ram_delta = peak_ram - rss_baseline
    results["TFT"] = {
        "train_time": t_train,
        "infer_time": t_infer,
        "ram_total_mb": peak_ram,
        "ram_delta_mb": ram_delta,
        "peak_vram_mb": peak_vram,
    }
    print(f"  Train: {fmt_secs(t_train)}, Infer: {t_infer*1000:.1f} ms/win, "
          f"RAM: {peak_ram:.0f} MB  (+{ram_delta:.0f} MB vs baseline), "
          f"VRAM: {peak_vram:.0f} MB\n")

    # ---- Chronos-2 ----
    print("Profiling Chronos-2 (zero-shot) ...")
    load_time, t_infer, peak_ram, peak_vram = profile_chronos(
        train_df, test_df, field_id
    )
    ram_delta = peak_ram - rss_baseline
    results["Chronos-2"] = {
        "load_time": load_time,
        "infer_time": t_infer,
        "ram_total_mb": peak_ram,
        "ram_delta_mb": ram_delta,
        "peak_vram_mb": peak_vram,
    }
    print(f"  Load: {fmt_secs(load_time)}, Infer: {t_infer*1000:.1f} ms/win, "
          f"RAM: {peak_ram:.0f} MB  (+{ram_delta:.0f} MB vs baseline), "
          f"VRAM: {peak_vram:.0f} MB\n")

    # ---- Markdown table ----
    print("=" * 85)
    print("SYSTEM EFFICIENCY BENCHMARKS  (2-year dataset, 724 train / 7 test days)")
    print("=" * 85)
    print()
    print(f"Baseline RSS (Python + data): {rss_baseline:.0f} MB")
    print()
    print("| Model | Training | Inference | RAM delta vs baseline | Total RSS | Peak VRAM |")
    print("|-------|----------|-----------|----------------------|-----------|-----------|")
    for model in ["N-HiTS", "TFT", "Chronos-2"]:
        r = results[model]
        if model == "Chronos-2":
            train_str = f"Zero-shot (load {fmt_secs(r['load_time'])})"
        else:
            train_str = fmt_secs(r["train_time"])
        print(f"| {model} | {train_str} | "
              f"{r['infer_time']*1000:.1f} ms/win | "
              f"+{r['ram_delta_mb']:.0f} MB | "
              f"{r['ram_total_mb']:.0f} MB | "
              f"{r['peak_vram_mb']:.0f} MB |")
    print()
    print("_Inference latency averaged over 5 runs after 2 warm-up runs. "
          "RAM measured as process RSS during inference._")


if __name__ == "__main__":
    main()
