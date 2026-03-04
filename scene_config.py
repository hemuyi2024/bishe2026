"""
场景配置：从 H5 路径解析场景名，以及每个场景的 mapbox、tif 目录、uav_infos.csv 路径。
约定：H5 路径形如 .../RealUAV/<scene_name>/netvlad/db/... 或 .../netvlad/query/...
      数据集根目录形如 /home/lty/datasets/RealUAV/<scene_name>/mapbox.tif, tif/, uav_infos.csv
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

# 数据集根目录（与 H5 中的 RealUAV 对应，存放 mapbox.tif、tif/、uav_infos.csv）
DATASETS_BASE = Path("/home/lty/datasets/RealUAV")


def get_scene_name_from_h5_path(h5_path: Path) -> str:
    """
    从 H5 路径解析场景名。
    约定：.../RealUAV/<scene_name>/netvlad/db/... 或 .../netvlad/query/...
    即 parent.parent.parent 为场景目录名。
    """
    p = Path(h5_path).resolve()
    # .../scene_name/netvlad/db/global-feats-netvlad.h5 -> scene_name
    if "netvlad" in p.parts:
        idx = p.parts.index("netvlad")
        if idx > 0:
            return p.parts[idx - 1]
    # 兼容：最后一级父目录的父目录名
    return p.parent.parent.parent.name


def get_scene_paths(
    scene_name: str,
    datasets_base: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    返回该场景的 mapbox、tif 目录、uav_infos 路径。
    """
    base = Path(datasets_base) if datasets_base is not None else DATASETS_BASE
    scene_dir = base / scene_name
    return {
        "mapbox_path": scene_dir / "mapbox.tif",
        "tif_dir": scene_dir / f"tif_{scene_name}",
        "uav_infos_path": scene_dir / "uav_infos.csv",
    }


def get_db_scene_ranges(
    db_h5_paths: List[Path],
    db_names: List[str],
) -> List[tuple]:
    """
    根据 DB H5 顺序，得到每个场景在合并 db_names 中的 (start, end) 范围。
    db_names 与 db_h5_paths 顺序一致（先第 1 个 H5 的全部，再第 2 个 H5 的全部...）。
    返回: [(scene_name, start, end), ...]
    """
    from demo_readH5 import load_netvlad_descriptors

    ranges = []
    offset = 0
    for h5_path in db_h5_paths:
        names, _ = load_netvlad_descriptors(Path(h5_path))
        n = len(names)
        scene_name = get_scene_name_from_h5_path(Path(h5_path))
        ranges.append((scene_name, offset, offset + n))
        offset += n
    return ranges


def get_query_trajectory_ranges(
    query_h5_paths: List[Path],
) -> List[tuple]:
    """
    每个 Query H5 对应一条轨迹，返回每条轨迹在「合并 query 列表」中的 (start, end) 及场景名。
    返回: [(scene_name, start, end), ...]
    """
    from demo_readH5 import load_netvlad_descriptors

    ranges = []
    offset = 0
    for h5_path in query_h5_paths:
        names, _ = load_netvlad_descriptors(Path(h5_path))
        n = len(names)
        scene_name = get_scene_name_from_h5_path(Path(h5_path))
        ranges.append((scene_name, offset, offset + n))
        offset += n
    return ranges
