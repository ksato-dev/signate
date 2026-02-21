"""
FMA tracks.csv の EDA: label_master.csv の10クラスに相当する分布を集計する。
30秒以上あるトラックのみ対象（利用時は先頭30秒を使用する前提）。
"""
import pandas as pd
import numpy as np
import ast
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MIN_DURATION_SEC = 30  # この長さ以上のトラックのみ取り扱う（先頭30秒を使用）

label_master = pd.read_csv("data/label_master.csv")
tracks = pd.read_csv(
    "data/FMA-Free_Music_Archive-Small&Medium/fma_metadata/tracks.csv",
    header=[0, 1, 2],
    low_memory=False,
)

# duration でフィルタ（30秒以上のものだけ）
dur_col = [c for c in tracks.columns if c[1] == "duration"][0]
tracks = tracks[tracks[dur_col] >= MIN_DURATION_SEC].copy()
tracks.reset_index(drop=True, inplace=True)

# 列名: track.genre_top, track.genres_all
genre_top_col = [c for c in tracks.columns if c[1] == "genre_top"][0]
genres_all_col = [c for c in tracks.columns if c[1] == "genres_all"][0]

genre_top = tracks[genre_top_col]
genres_all_raw = tracks[genres_all_col]

# genres_all は "[21]" や "[76, 103]" の文字列
def parse_genre_ids(s):
    if pd.isna(s) or s == "":
        return []
    try:
        return list(ast.literal_eval(s))
    except Exception:
        return []

# FMA genre_id -> label_master label_name
# 優先: サブジャンルで disco/metal/reggae を先に判定
FMA_TO_LABEL = {
    # 直接対応 (genre_top または top-level genre_id)
    3: "blues",
    4: "jazz",
    5: "classical",
    9: "country",
    10: "pop",
    12: "rock",
    21: "hiphop",
    # サブジャンル
    11: "disco",
    31: "metal",
    79: "reggae",   # Reggae - Dub
    602: "reggae",  # Reggae - Dancehall
}

# genre_top 文字列 -> label_name
GENRE_TOP_TO_LABEL = {
    "Blues": "blues",
    "Classical": "classical",
    "Country": "country",
    "Disco": "disco",
    "Hip-Hop": "hiphop",
    "Jazz": "jazz",
    "Metal": "metal",
    "Pop": "pop",
    "Reggae": "reggae",
    "Rock": "rock",
}

def assign_label(row):
    """1トラックを label_master の1クラスに割り当て（優先: genres_all の disco/metal/reggae）"""
    ids = parse_genre_ids(row[genres_all_col])
    top = row[genre_top_col]

    # 1) genres_all に 11, 31, 79, 602 があればそれで判定
    for gid in [11, 31, 79, 602]:
        if gid in ids:
            return FMA_TO_LABEL[gid]

    # 2) genre_top で判定
    if pd.notna(top) and top in GENRE_TOP_TO_LABEL:
        return GENRE_TOP_TO_LABEL[top]

    # 3) genres_all の先頭の id を FMA_TO_LABEL でマッピング（Rock 下の Metal 等は genre_top が Rock のためここでは Rock）
    for gid in ids:
        if gid in FMA_TO_LABEL:
            return FMA_TO_LABEL[gid]

    return None  # 未対応

assigned = tracks.apply(assign_label, axis=1)

# 集計
print("=" * 60)
print("FMA tracks.csv × label_master クラス分布 (EDA)")
print("=" * 60)
print(f"対象: duration >= {MIN_DURATION_SEC} 秒のトラックのみ（利用時は先頭30秒を使用）")
print(f"総トラック数: {len(tracks):,}")
print(f"label_master に割り当てた数: {assigned.notna().sum():,}")
print(f"未対応 (その他): {(assigned.isna()).sum():,}")
print()

counts = assigned.value_counts(dropna=False).sort_index()
label_order = list(label_master["label_name"].values)
for name in label_order:
    n = counts.get(name, 0)
    pct = 100 * n / assigned.notna().sum() if assigned.notna().any() else 0
    print(f"  {name:<12} {n:>8,} 件  ({pct:>5.1f}%)")

if pd.isna(assigned).any():
    n_other = assigned.isna().sum()
    pct_other = 100 * n_other / len(tracks)
    print(f"  {'(未対応)':<12} {n_other:>8,} 件  (全件比 {pct_other:.1f}%)")

print()
print("--- genre_top のみの分布（参考） ---")
top_counts = genre_top.value_counts(dropna=False)
for t, c in top_counts.head(20).items():
    label = GENRE_TOP_TO_LABEL.get(t, "-")
    print(f"  {str(t):<25} {c:>8,}  -> {label}")
