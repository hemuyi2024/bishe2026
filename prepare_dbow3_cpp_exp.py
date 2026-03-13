#!/usr/bin/env python3
"""
为 C++ DBoW3 检索实验准备输入文件。

生成到 output/dbow3_exp/ 目录：
- db_list.txt: DB 图像路径，每行一条，行号=db_index
- query_list.txt: Query 图像路径，每行一条，行号=query_index
- gt.txt: 每 query 的 GT DB 索引集合，格式 "query_idx: idx1,idx2,..."（0-based）
- eval_mask.txt: 每行 0 或 1，1=参与评估（排除 industry2、rural2）

用法: python prepare_dbow3_cpp_exp.py
"""
import sys
from pathlib import Path

# 确保能导入 ch3 模块
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from demo_readH5 import load_multi_netvlad_descriptors, DB_H5_PATHS, QUERY_H5_PATHS
from scene_config import get_db_scene_ranges, get_query_trajectory_ranges, DATASETS_BASE, get_scene_paths
from coords_gt_utils import (
    build_gt_9_for_all_trajectories,
    build_database_coords,
    get_db_coords_for_scene,
    get_geotransform_and_srs,
    load_uav_infos,
    lonlat_to_pixel,
)
from bow_retrieval import get_all_image_paths

import numpy as np


def main():
    out_dir = _script_dir / "output" / "dbow3_exp"
    out_dir.mkdir(parents=True, exist_ok=True)

    db_names, _ = load_multi_netvlad_descriptors(DB_H5_PATHS)
    query_names, _ = load_multi_netvlad_descriptors(QUERY_H5_PATHS)
    db_scene_ranges = get_db_scene_ranges(DB_H5_PATHS, db_names)
    trajectory_ranges = get_query_trajectory_ranges(QUERY_H5_PATHS)

    db_paths = get_all_image_paths(db_names, db_scene_ranges)
    query_paths = get_all_image_paths(query_names, trajectory_ranges)

    gt_9_list, _ = build_gt_9_for_all_trajectories(
        trajectory_ranges, db_scene_ranges, db_names,
        datasets_base=DATASETS_BASE, verbose=False
    )

    # valid_for_metrics: 排除 industry2, rural2
    n_queries = len(query_names)
    exclude_scenes = {"industry2", "rural2"}
    exclude_mask = [False] * n_queries
    for scene_name, start, end in trajectory_ranges:
        if scene_name in exclude_scenes:
            for i in range(start, end):
                exclude_mask[i] = True

    valid = [len(gt_9_list[i]) > 0 for i in range(n_queries)]
    eval_mask = [1 if (valid[i] and not exclude_mask[i]) else 0 for i in range(n_queries)]

    # 写入 db_list.txt
    with (out_dir / "db_list.txt").open("w", encoding="utf-8") as f:
        for p in db_paths:
            f.write(str(p) + "\n")

    # 写入 query_list.txt
    with (out_dir / "query_list.txt").open("w", encoding="utf-8") as f:
        for p in query_paths:
            f.write(str(p) + "\n")

    # 写入 gt.txt
    with (out_dir / "gt.txt").open("w", encoding="utf-8") as f:
        for i, gt_set in enumerate(gt_9_list):
            idxs = sorted(gt_set)
            line = f"{i}:" + ",".join(str(x) for x in idxs) + "\n"
            f.write(line)

    # 写入 eval_mask.txt
    with (out_dir / "eval_mask.txt").open("w", encoding="utf-8") as f:
        for v in eval_mask:
            f.write(str(v) + "\n")

    # 写入 query_scene.txt: 每行一个 scene_name，与 query 一一对应
    query_scene = []
    for scene_name, start, end in trajectory_ranges:
        query_scene.extend([scene_name] * (end - start))
    with (out_dir / "query_scene.txt").open("w", encoding="utf-8") as f:
        for s in query_scene:
            f.write(s + "\n")

    # 写入 db_coords_proj.txt, query_coords_proj.txt: 投影坐标 (x, y) 米，用于 Mean localization error
    base = Path(DATASETS_BASE)
    db_scene_by_name = {name: (s, e) for name, s, e in db_scene_ranges}
    db_coords_px = build_database_coords(
        db_names, db_scene_ranges, datasets_base=base, verbose=False
    )
    db_coords_proj = np.full((len(db_names), 2), np.nan)
    for scene_name, start, end in db_scene_ranges:
        paths = get_scene_paths(scene_name, base)
        gt_tuple, _ = get_geotransform_and_srs(paths["mapbox_path"])
        if gt_tuple is None:
            continue
        gt = gt_tuple
        for i in range(start, end):
            px, py = db_coords_px[i, 0], db_coords_px[i, 1]
            if np.isnan(px) or np.isnan(py):
                continue
            x_geo = gt[0] + gt[1] * px + gt[2] * py
            y_geo = gt[3] + gt[4] * px + gt[5] * py
            db_coords_proj[i, 0], db_coords_proj[i, 1] = x_geo, y_geo

    query_coords_proj = np.full((n_queries, 2), np.nan)
    for scene_name, q_start, q_end in trajectory_ranges:
        if scene_name not in db_scene_by_name:
            continue
        paths = get_scene_paths(scene_name, base)
        uav_path = paths["uav_infos_path"]
        mapbox_path = paths["mapbox_path"]
        if not uav_path.exists() or not mapbox_path.exists():
            continue
        gt_tuple, _ = get_geotransform_and_srs(mapbox_path)
        if gt_tuple is None:
            continue
        lats, lons = load_uav_infos(uav_path)
        n_use = min(len(lats), q_end - q_start)
        for i in range(n_use):
            px, py, x_geo, y_geo = lonlat_to_pixel(
                lons[i], lats[i], gt_tuple, None, return_geo=True
            )
            query_coords_proj[q_start + i, 0] = x_geo
            query_coords_proj[q_start + i, 1] = y_geo

    with (out_dir / "db_coords_proj.txt").open("w", encoding="utf-8") as f:
        for i in range(len(db_names)):
            x, y = db_coords_proj[i, 0], db_coords_proj[i, 1]
            if np.isnan(x) or np.isnan(y):
                f.write("nan nan\n")
            else:
                f.write(f"{x} {y}\n")
    with (out_dir / "query_coords_proj.txt").open("w", encoding="utf-8") as f:
        for i in range(n_queries):
            x, y = query_coords_proj[i, 0], query_coords_proj[i, 1]
            if np.isnan(x) or np.isnan(y):
                f.write("nan nan\n")
            else:
                f.write(f"{x} {y}\n")

    n_eval = sum(eval_mask)
    print(f"已生成到 {out_dir}")
    print(f"  DB: {len(db_paths)} 条")
    print(f"  Query: {len(query_paths)} 条")
    print(f"  参与评估 query 数（排除 industry2/rural2）: {n_eval}")


if __name__ == "__main__":
    main()
