"""
リーク検出スクリプト
GTZAN / FMA / Train / Test 間の音声類似度を log-mel 埋め込みのコサイン類似度で検出する。
上位 N ペアを表示し、同一音源の混入がないか確認する。

埋め込みは .npy キャッシュに保存し、再実行時はキャッシュから読み込む。
"""

import os
import time
import warnings
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

warnings.simplefilter('ignore')

# ==============================================================================
# 設定
# ==============================================================================
SR = 22050
N_MELS = 128
DURATION = 30  # 秒
TOP_K = 30     # 表示する上位ペア数
N_WORKERS = 8  # 並列ワーカー数

DATA_DIR = Path("data")
CACHE_DIR = Path("data/leak_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIRS = [
    DATA_DIR / "train_sound_1" / "train_sound_1",
    DATA_DIR / "train_sound_2" / "train_sound_2",
    DATA_DIR / "train_sound_3" / "train_sound_3",
]

TEST_DIRS = [
    DATA_DIR / "test_sound_1" / "test_sound_1",
    DATA_DIR / "test_sound_2" / "test_sound_2",
    DATA_DIR / "test_sound_3" / "test_sound_3",
]

GTZAN_DIR = DATA_DIR / "GTZAN_Dataset" / "Data" / "genres_original"
GTZAN_GENRE_DIRS = sorted(GTZAN_DIR.iterdir()) if GTZAN_DIR.exists() else []

FMA_BASE = DATA_DIR / "FMA-Free_Music_Archive-Small&Medium"
FMA_SUBSETS = ["fma_small", "fma_medium"]


def collect_files(dirs, extensions=("*.au", "*.wav", "*.mp3")):
    files = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for ext in extensions:
            files.extend(sorted(d.glob(ext)))
    return files


def collect_fma_all():
    """FMA 全 mp3 を再帰収集"""
    files = []
    for subset in FMA_SUBSETS:
        subset_dir = FMA_BASE / subset / subset
        if not subset_dir.exists():
            continue
        files.extend(sorted(subset_dir.rglob("*.mp3")))
    return files


def _extract_one(path_str):
    """1ファイルの埋め込みを抽出（並列ワーカー用）"""
    try:
        y, _ = librosa.load(path_str, sr=SR, duration=DURATION, mono=True)
    except Exception:
        return path_str, None

    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    emb = np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])
    return path_str, emb


def extract_embedding_full(path, sr=SR, n_mels=N_MELS, duration=DURATION):
    try:
        y, _ = librosa.load(str(path), sr=sr, duration=duration, mono=True)
    except Exception:
        return None
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max).flatten()


def build_embeddings_cached(files, label, cache_name, parallel=False):
    """埋め込み抽出 + キャッシュ"""
    emb_cache = CACHE_DIR / f"{cache_name}_emb.npy"
    paths_cache = CACHE_DIR / f"{cache_name}_paths.npy"

    if emb_cache.exists() and paths_cache.exists():
        emb = np.load(emb_cache)
        paths = np.load(paths_cache, allow_pickle=True).tolist()
        if len(emb) == len(paths) and len(emb) > 0:
            print(f"\n[{label}] Cache hit: {len(emb)} embeddings from {cache_name}")
            return emb, paths

    total = len(files)
    print(f"\n[{label}] Extracting embeddings from {total} files"
          f"{' (parallel)' if parallel else ''}...")
    t0 = time.time()
    embeddings, paths, failed = [], [], 0

    if parallel and total > 100:
        path_strs = [str(f) for f in files]
        done = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_extract_one, p): p for p in path_strs}
            for future in as_completed(futures):
                p, emb = future.result()
                done += 1
                if done % 500 == 0 or done == total:
                    elapsed = time.time() - t0
                    speed = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / speed if speed > 0 else 0
                    print(f"  {done:>6d}/{total}  ({speed:.1f} files/s, ETA {eta:.0f}s)")
                if emb is not None:
                    embeddings.append(emb)
                    paths.append(p)
                else:
                    failed += 1
    else:
        for i, f in enumerate(files):
            if (i + 1) % 200 == 0 or i == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / speed if speed > 0 else 0
                print(f"  {i+1:>6d}/{total}  ({speed:.1f} files/s, ETA {eta:.0f}s)")
            _, emb = _extract_one(str(f))
            if emb is not None:
                embeddings.append(emb)
                paths.append(str(f))
            else:
                failed += 1

    elapsed = time.time() - t0
    print(f"  Done: {len(embeddings)} embeddings in {elapsed:.1f}s (failed: {failed})")

    emb_arr = np.array(embeddings)
    np.save(emb_cache, emb_arr)
    np.save(paths_cache, np.array(paths, dtype=object))
    return emb_arr, paths


def find_top_pairs(emb_a, paths_a, emb_b, paths_b, top_k=TOP_K):
    """各 source の最近傍 target を取り、上位 top_k を返す"""
    sim_matrix = cosine_similarity(emb_a, emb_b)
    pairs = []
    for i in range(sim_matrix.shape[0]):
        j = int(np.argmax(sim_matrix[i]))
        pairs.append((float(sim_matrix[i, j]), paths_a[i], paths_b[j]))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:top_k]


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_section("リーク検出: GTZAN / FMA / Train <-> Test")

    train_files = collect_files(TRAIN_DIRS)
    test_files = collect_files(TEST_DIRS)
    gtzan_files = collect_files(GTZAN_GENRE_DIRS, extensions=("*.wav",))
    fma_files = collect_fma_all()

    print(f"\n  Train:  {len(train_files)}")
    print(f"  Test:   {len(test_files)}")
    print(f"  GTZAN:  {len(gtzan_files)}")
    print(f"  FMA:    {len(fma_files)}")

    if not test_files:
        print("ERROR: No test files found.")
        return

    # --- Embeddings ---
    print_section("Phase 1: 粗い埋め込み (log-mel mean+std) でスクリーニング")

    test_emb, test_paths = build_embeddings_cached(test_files, "Test", "test")

    all_pairs = {}

    if gtzan_files:
        gtzan_emb, gtzan_paths = build_embeddings_cached(gtzan_files, "GTZAN", "gtzan")
        pairs = find_top_pairs(gtzan_emb, gtzan_paths, test_emb, test_paths, TOP_K)
        all_pairs["GTZAN"] = pairs
        print(f"\n--- GTZAN -> Test (Top {TOP_K}) ---")
        for r, (s, a, b) in enumerate(pairs, 1):
            print(f"  #{r:2d}  sim={s:.6f}  {Path(a).name:<30s} <-> {Path(b).name}")

    if fma_files:
        fma_emb, fma_paths = build_embeddings_cached(
            fma_files, "FMA", "fma_all", parallel=True)
        pairs = find_top_pairs(fma_emb, fma_paths, test_emb, test_paths, TOP_K)
        all_pairs["FMA"] = pairs
        print(f"\n--- FMA -> Test (Top {TOP_K}) ---")
        for r, (s, a, b) in enumerate(pairs, 1):
            print(f"  #{r:2d}  sim={s:.6f}  {Path(a).name:<30s} <-> {Path(b).name}")

    train_emb, train_paths = build_embeddings_cached(train_files, "Train", "train")
    pairs = find_top_pairs(train_emb, train_paths, test_emb, test_paths, TOP_K)
    all_pairs["Train"] = pairs
    print(f"\n--- Train -> Test (Top {TOP_K}) ---")
    for r, (s, a, b) in enumerate(pairs, 1):
        print(f"  #{r:2d}  sim={s:.6f}  {Path(a).name:<30s} <-> {Path(b).name}")

    # --- Phase 2: full mel 再検証 ---
    threshold = 0.98
    suspect = []
    for src_type, pairs in all_pairs.items():
        suspect += [(s, a, b, f"{src_type}->Test") for s, a, b in pairs if s >= threshold]

    if suspect:
        print_section(f"Phase 2: 高類似度 ({len(suspect)}件, sim>={threshold}) を full mel 再検証")
        for sim_c, src, tgt, ptype in suspect:
            e_src = extract_embedding_full(src)
            e_tgt = extract_embedding_full(tgt)
            if e_src is not None and e_tgt is not None:
                sim_f = float(cosine_similarity(
                    e_src.reshape(1, -1), e_tgt.reshape(1, -1))[0, 0])
                flag = (" *** LEAK ***" if sim_f >= 0.999
                        else " ** VERY SIMILAR **" if sim_f >= 0.99 else "")
                print(f"  [{ptype}] coarse={sim_c:.6f}  full={sim_f:.6f}"
                      f"  {Path(src).name} <-> {Path(tgt).name}{flag}")

    # --- サマリー ---
    print_section("サマリー")
    csv_rows = []
    for src_type, pairs in all_pairs.items():
        if not pairs:
            continue
        max_s = pairs[0][0]
        n99 = sum(1 for s, _, _ in pairs if s >= 0.99)
        n999 = sum(1 for s, _, _ in pairs if s >= 0.999)
        print(f"  {src_type:>8s} <-> Test  max={max_s:.6f}  sim>=0.999: {n999}  sim>=0.99: {n99}")
        for s, a, b in pairs:
            csv_rows.append((src_type, s, Path(a).name, Path(b).name))

    leak_sources = [(t, sum(1 for s, _, _ in p if s >= 0.999))
                     for t, p in all_pairs.items()
                     if sum(1 for s, _, _ in p if s >= 0.999) > 0]
    safe_sources = [t for t, p in all_pairs.items()
                    if p and p[0][0] < 0.99]

    if leak_sources:
        print()
        for t, n in leak_sources:
            print(f"  !! {t}: {n}件の同一音源検出 → リーク確定")
        print("    → これらを学習に含めると Private LB で崩壊リスク大")
    if safe_sources:
        print(f"\n  安全な外部データ候補: {', '.join(safe_sources)}")

    output_csv = "leak_detection_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "source_type", "similarity", "source_file", "target_file"])
        for i, (t, s, a, b) in enumerate(csv_rows, 1):
            w.writerow([i, t, f"{s:.6f}", a, b])
    print(f"\n  結果を {output_csv} に保存しました。")


if __name__ == "__main__":
    main()
