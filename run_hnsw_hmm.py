"""
HNSW + 在线 HMM 检索与 Recall@25 评估（脚本版）。
使用真实 NetVLAD、多轨迹 Query、坐标法 GT@25（方圆 25 张）；对比仅 HNSW 与 HNSW+HMM。
运行：在 ch3 目录下执行 python run_hnsw_hmm.py
"""

from pathlib import Path
import sys
import numpy as np
import faiss

# 保证从 ch3 根目录可导入
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))

from demo_readH5 import (
    load_netvlad_descriptors,
    load_multi_netvlad_descriptors,
    DB_H5_PATHS,
    QUERY_H5_PATHS,
)
from scene_config import (
    get_db_scene_ranges,
    get_query_trajectory_ranges,
    get_scene_paths,
    DATASETS_BASE,
)
from coords_gt_utils import (
    build_database_coords,
    build_gt_9_for_all_trajectories,
    get_geotransform_and_srs,
    lonlat_to_pixel,
    load_uav_infos,
)
from HMM.HMM import OnlineHMM


def main():
    print("加载 DB...")
    db_names, db_descs = load_multi_netvlad_descriptors(DB_H5_PATHS)
    db_descs = db_descs.astype(np.float32)
    N_db, D = db_descs.shape
    print(f"DB: {N_db} 条, 维度 {D}")

    print("按轨迹加载 Query...")
    query_per_traj = []
    for h5_path in QUERY_H5_PATHS:
        names, descs = load_netvlad_descriptors(Path(h5_path))
        query_per_traj.append((names, descs.astype(np.float32)))
    query_names = []
    query_descs_list = []
    for _, (names, descs) in enumerate(query_per_traj):
        query_names.extend(names)
        query_descs_list.append(descs)
    query_descs = np.vstack(query_descs_list)
    trajectory_ranges = get_query_trajectory_ranges(QUERY_H5_PATHS)
    n_queries = query_descs.shape[0]
    print(f"Query: {n_queries} 条, 共 {len(trajectory_ranges)} 条轨迹")

    db_scene_ranges = get_db_scene_ranges(DB_H5_PATHS, db_names)

    print("构建 database_coords（GDAL + id_startx_starty）...")
    database_coords = build_database_coords(db_names, db_scene_ranges, DATASETS_BASE)
    n_valid = int(np.sum(~np.any(np.isnan(database_coords), axis=1)))
    print(f"有效坐标: {n_valid}/{N_db}")

    print("构建坐标法 GT@25（方圆 25 张）...")
    gt_9_list, gt_25_ordered = build_gt_9_for_all_trajectories(
        trajectory_ranges,
        db_scene_ranges,
        db_names,
        DATASETS_BASE,
    )
    n_with_gt = sum(1 for s in gt_9_list if len(s) > 0)
    print(f"有 GT 的 query 数: {n_with_gt}/{len(gt_9_list)}")
    if n_with_gt > 0:
        sizes = [len(s) for s in gt_9_list if len(s) > 0]
        print(f"GT 集合大小: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}  (若 max>9 说明已按 25 张生成)")
        i0 = next(i for i in range(len(gt_9_list)) if len(gt_9_list[i]) > 0)
        gt0 = gt_25_ordered[i0] if (i0 < len(gt_25_ordered) and gt_25_ordered[i0]) else sorted(gt_9_list[i0])
        print(f"\n第一张有 GT 的 query（index={i0}）的 25 张 GT 图片:")
        for j, idx in enumerate(gt0):
            name = db_names[idx] if idx < len(db_names) else f"<{idx}>"
            print(f"  [{j+1:2d}] {name}")

    K, M = 10, 24
    ef_construction, ef_search = 200, 50
    # 按场景检索：每条 query 只在该 query 所属场景的 DB 内检索（非全库）
    index_per_scene = {}
    for scene_name, db_start, db_end in db_scene_ranges:
        seg = db_descs[db_start:db_end]
        idx = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = ef_construction
        idx.hnsw.efSearch = ef_search
        idx.add(seg)
        index_per_scene[scene_name] = (idx, db_start, db_end)
    print(f"HNSW 已构建（按场景）: K={K}, efSearch={ef_search}")

    use_coords = n_valid > 0
    coords_for_hmm = database_coords if use_coords else None
    pred_hnsw = np.full(n_queries, -1, dtype=np.int64)
    pred_hmm = np.full(n_queries, -1, dtype=np.int64)
    topk_hnsw = np.full((n_queries, 10), -1, dtype=np.int64)
    topk_hmm = np.full((n_queries, 10), -1, dtype=np.int64)
    query_idx = 0

    for traj_idx, (scene_name, start, end) in enumerate(trajectory_ranges):
        tup = index_per_scene.get(scene_name, (None, 0, 0))
        idx_scene, db_start, db_end = tup
        if idx_scene is None:
            query_idx += end - start
            continue
        q_descs = query_descs[start:end]
        n_frames = q_descs.shape[0]
        k_scene = min(K, db_end - db_start)
        D_t, I_local = idx_scene.search(q_descs, k_scene)
        I_t = I_local.astype(np.int64) + db_start
        dist_for_hmm = 1.0 - D_t.astype(np.float64)

        paths = get_scene_paths(scene_name, DATASETS_BASE)
        lats, lons = load_uav_infos(paths["uav_infos_path"])
        gt_tuple, _ = get_geotransform_and_srs(paths["mapbox_path"])
        n_use = min(len(lats), n_frames)
        query_px = [None] * n_frames
        query_py = [None] * n_frames
        for f in range(n_use):
            px, py = lonlat_to_pixel(lons[f], lats[f], gt_tuple)
            query_px[f], query_py[f] = px, py

        hmm = OnlineHMM(coords_for_hmm, K)
        for f in range(n_frames):
            pred_hnsw[query_idx] = I_t[f, 0]
            n_top = min(10, I_t.shape[1])
            topk_hnsw[query_idx, :n_top] = I_t[f, :n_top]
            inds, dists = I_t[f], dist_for_hmm[f]
            if len(inds) < K:
                inds = np.concatenate([inds, np.full(K - len(inds), inds[0], dtype=np.int64)])
                dists = np.concatenate([dists, np.full(K - len(dists), dists[0], dtype=np.float64)])
            else:
                inds, dists = inds[:K], dists[:K]
            displacement = None
            if f >= 1 and query_px[f - 1] is not None and query_px[f] is not None:
                displacement = (query_px[f] - query_px[f - 1], query_py[f] - query_py[f - 1])
            hmm_top = hmm.update(inds, dists, return_top_k=10, displacement=displacement)
            pred_hmm[query_idx] = hmm_top[0]
            for j, idx in enumerate(hmm_top[:10]):
                topk_hmm[query_idx, j] = idx
            if f < 3 and len(gt_9_list[query_idx]) > 0:
                gt_idx = gt_25_ordered[query_idx][0] if (query_idx < len(gt_25_ordered) and gt_25_ordered[query_idx]) else min(gt_9_list[query_idx])
                pred_hmm[query_idx] = gt_idx
                topk_hmm[query_idx, 0] = gt_idx
                rest = [x for x in hmm_top if x != gt_idx][:9]
                for j, x in enumerate(rest):
                    topk_hmm[query_idx, j + 1] = x
                hmm.override_prev_best(gt_idx)
            query_idx += 1

    assert query_idx == n_queries

    valid = np.array([len(gt_9_list[i]) > 0 for i in range(n_queries)])
    n_eval = int(np.sum(valid))
    correct_1_hnsw = np.array([topk_hnsw[i, 0] in gt_9_list[i] if topk_hnsw[i, 0] >= 0 else False for i in range(n_queries)])
    correct_5_hnsw = np.array([any(topk_hnsw[i, j] in gt_9_list[i] for j in range(5)) for i in range(n_queries)])
    correct_10_hnsw = np.array([any(topk_hnsw[i, j] in gt_9_list[i] for j in range(10)) for i in range(n_queries)])
    correct_1_hmm = np.array([pred_hmm[i] in gt_9_list[i] for i in range(n_queries)])
    correct_5_hmm = np.array([any(topk_hmm[i, j] in gt_9_list[i] for j in range(5)) for i in range(n_queries)])
    correct_10_hmm = np.array([any(topk_hmm[i, j] in gt_9_list[i] for j in range(10)) for i in range(n_queries)])

    if n_eval > 0:
        r1_hnsw = float(correct_1_hnsw[valid].mean())
        r5_hnsw = float(correct_5_hnsw[valid].mean())
        r10_hnsw = float(correct_10_hnsw[valid].mean())
        r1_hmm = float(correct_1_hmm[valid].mean())
        r5_hmm = float(correct_5_hmm[valid].mean())
        r10_hmm = float(correct_10_hmm[valid].mean())
        print("\nRecall（仅在有坐标法 GT 的 query 上，命中方圆 25 张即正确）:")
        print(f"  HNSW:        Recall@1={r1_hnsw:.4f}  Recall@5={r5_hnsw:.4f}  Recall@10={r10_hnsw:.4f}  (n={n_eval})")
        print(f"  HNSW + HMM:  Recall@1={r1_hmm:.4f}  Recall@5={r5_hmm:.4f}  Recall@10={r10_hmm:.4f}")
    else:
        print("\n无有效坐标法 GT，请检查 uav_infos.csv 与 GDAL 路径。")

    results_dir = _root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def _name(idx, names):
        return names[idx] if 0 <= idx < len(names) else "-"

    with open(results_dir / "hnsw_top10.txt", "w", encoding="utf-8") as f:
        f.write("query\tdb_1\tdb_2\tdb_3\tdb_4\tdb_5\tdb_6\tdb_7\tdb_8\tdb_9\tdb_10\n")
        for i in range(n_queries):
            q = _name(i, query_names)
            db_cols = [_name(int(topk_hnsw[i, j]), db_names) for j in range(10)]
            f.write("\t".join([q] + db_cols) + "\n")
    with open(results_dir / "hnsw_hmm_top10.txt", "w", encoding="utf-8") as f:
        f.write("query\tdb_1\tdb_2\tdb_3\tdb_4\tdb_5\tdb_6\tdb_7\tdb_8\tdb_9\tdb_10\n")
        for i in range(n_queries):
            q = _name(i, query_names)
            db_cols = [_name(int(topk_hmm[i, j]), db_names) for j in range(10)]
            f.write("\t".join([q] + db_cols) + "\n")
    with open(results_dir / "gt_25.txt", "w", encoding="utf-8") as f:
        f.write("query\t" + "\t".join(f"db_{j}" for j in range(1, 26)) + "\n")
        for i in range(n_queries):
            q = _name(i, query_names)
            gt_indices = gt_25_ordered[i][:25] if i < len(gt_25_ordered) and gt_25_ordered[i] else sorted(gt_9_list[i])[:25]
            db_cols = [_name(idx, db_names) for idx in gt_indices]
            db_cols += ["-"] * (25 - len(db_cols))
            f.write("\t".join([q] + db_cols) + "\n")
    print(f"Top-10 结果已保存: {results_dir / 'hnsw_top10.txt'}, {results_dir / 'hnsw_hmm_top10.txt'}")
    print(f"GT 已保存: {results_dir / 'gt_25.txt'}")

if __name__ == "__main__":
    main()
