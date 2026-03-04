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
    DATASETS_BASE,
)
from coords_gt_utils import (
    build_database_coords,
    build_gt_9_for_all_trajectories,
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
    gt_9_list = build_gt_9_for_all_trajectories(
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
        gt0 = sorted(gt_9_list[i0])
        print(f"\n第一张有 GT 的 query（index={i0}）的 25 张 GT 图片:")
        for j, idx in enumerate(gt0):
            name = db_names[idx] if idx < len(db_names) else f"<{idx}>"
            print(f"  [{j+1:2d}] {name}")

    K, M = 10, 24
    ef_construction, ef_search = 200, 50
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
        hmm = OnlineHMM(coords_for_hmm, K)
        for f in range(n_frames):
            pred_hnsw[query_idx] = I_t[f, 0]
            inds, dists = I_t[f], dist_for_hmm[f]
            if len(inds) < K:
                inds = np.concatenate([inds, np.full(K - len(inds), inds[0], dtype=np.int64)])
                dists = np.concatenate([dists, np.full(K - len(dists), dists[0], dtype=np.float64)])
            else:
                inds, dists = inds[:K], dists[:K]
            pred_hmm[query_idx] = hmm.update(inds, dists)
            query_idx += 1

    assert query_idx == n_queries

    correct_hnsw = np.array([pred_hnsw[i] in gt_9_list[i] for i in range(n_queries)])
    correct_hmm = np.array([pred_hmm[i] in gt_9_list[i] for i in range(n_queries)])
    valid = np.array([len(gt_9_list[i]) > 0 for i in range(n_queries)])
    n_eval = int(np.sum(valid))

    if n_eval > 0:
        recall_hnsw = float(correct_hnsw[valid].mean())
        recall_hmm = float(correct_hmm[valid].mean())
        print("\nRecall@25（仅在有坐标法 GT 的 query 上，命中方圆 25 张即正确）:")
        print(f"  仅 HNSW:      {recall_hnsw:.4f}  (n={n_eval})")
        print(f"  HNSW + HMM:   {recall_hmm:.4f}")
    else:
        print("\n无有效坐标法 GT，请检查 uav_infos.csv 与 GDAL 路径。")


if __name__ == "__main__":
    main()
