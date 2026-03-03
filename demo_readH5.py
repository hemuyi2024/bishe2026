import h5py
import numpy as np
from pathlib import Path


# 单一 DB / Query H5（保留，方便兼容原来的脚本）
DB_H5_PATH = Path("/home/lty/outputs/RealUAV/city3/netvlad/db/global-feats-netvlad.h5")
QUERY_H5_PATH = Path("/home/lty/outputs/RealUAV/city3/netvlad/query/global-feats-netvlad.h5")

# 多个 DB H5，可以在这里集中配置
DB_H5_PATHS = [
    Path("/home/lty/outputs/RealUAV/city1/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/city2/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/city3/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/village1/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/village2/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/rural1/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/rural2/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/rural3/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/school/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/suburbs1/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/suburbs2/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/industry1/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/industry2/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/industry3/netvlad/db/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/park1/netvlad/db/global-feats-netvlad.h5"),
]

# 多个 Query H5，结构与 DB_H5_PATHS 类似，方便在实验里统一合并
QUERY_H5_PATHS = [
    Path("/home/lty/outputs/RealUAV/city3/netvlad/query/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/industry1/netvlad/query/global-feats-netvlad.h5"),
    Path("/home/lty/outputs/RealUAV/park1/netvlad/query/global-feats-netvlad.h5"),
]


def inspect_h5_structure(h5_path: Path) -> None:
    """打印 H5 文件内部的层次结构和各数据集信息。"""
    print(f"\n=== 文件结构: {h5_path} ===")
    with h5py.File(h5_path, "r") as f:
        def _print(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group   : {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset : {name} - shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(_print)


def load_netvlad_descriptors(h5_path: Path):
    """
    从 hloc 生成的 NetVLAD H5 文件中读取：
    - img_names: 每个 global_descriptor 对应的图像键（例如 'seu_tif_m300/10_3000_1500.tif'）
    - descriptors: 形状为 (N, D) 的 float32 数组
    """
    img_names = []
    desc_list = []

    with h5py.File(h5_path, "r") as f:
        def _collect(name, obj):
            # 匹配所有以 'global_descriptor' 结尾的数据集
            if isinstance(obj, h5py.Dataset) and name.endswith("global_descriptor"):
                # name 形如 'seu_tif_m300/10_3000_1500.tif/global_descriptor'
                img_key = name.rsplit("/", 1)[0]
                img_names.append(img_key)
                # 转成 float32 以便后续给 Faiss 使用
                desc_list.append(obj[()].astype("float32"))

        f.visititems(_collect)

    if not desc_list:
        raise RuntimeError(f"在 {h5_path} 中没有找到任何 'global_descriptor' 数据集。")

    descriptors = np.stack(desc_list, axis=0)
    return img_names, descriptors


def load_multi_netvlad_descriptors(h5_paths):
    """
    从多个 NetVLAD H5 文件中读取并合并：
    - all_names: 合并后的图像键列表
    - all_descs: shape (N_total, D) 的 float32 数组
    """
    all_names = []
    all_descs = []

    for h5_path in h5_paths:
        names, descs = load_netvlad_descriptors(Path(h5_path))
        all_names.extend(names)
        all_descs.append(descs.astype("float32"))

    if not all_descs:
        raise RuntimeError("DB H5 路径列表为空，或者所有文件都没有找到 global_descriptor。")

    all_descs = np.vstack(all_descs)
    return all_names, all_descs


def main():
    # 1. 打印两个 H5 文件的结构
    inspect_h5_structure(DB_H5_PATH)
    inspect_h5_structure(QUERY_H5_PATH)

    # 2. 读取数据库和查询的 NetVLAD 描述子
    print("\n=== 读取数据库 NetVLAD 描述子（单一 DB_H5_PATH） ===")
    db_names, db_descs = load_netvlad_descriptors(DB_H5_PATH)
    print(f"DB(single): 共 {len(db_names)} 条，描述子形状 {db_descs.shape}")  # (N_db, D)

    print("\n=== 读取查询 NetVLAD 描述子 ===")
    query_names, query_descs = load_netvlad_descriptors(QUERY_H5_PATH)
    print(f"Query: 共 {len(query_names)} 条，描述子形状 {query_descs.shape}")  # (N_q, D)

    # 3. 示例：打印前几条数据，方便你确认内容
    preview = min(3, len(db_names))
    print(f"\n=== 数据库前 {preview} 条示例 ===")
    for i in range(preview):
        print(f"[{i}] name = {db_names[i]}")
        print(f"     descriptor[0:5] = {db_descs[i][:5]}")  # 只看前 5 维

    preview_q = min(3, len(query_names))
    print(f"\n=== 查询前 {preview_q} 条示例 ===")
    for i in range(preview_q):
        print(f"[{i}] name = {query_names[i]}")
        print(f"     descriptor[0:5] = {query_descs[i][:5]}")

    # 4. 这里不做 HNSW 索引，只负责把“索引、图片名字、向量”准备好
    #    后面你可以在其他脚本中直接导入 load_netvlad_descriptors 来构建 Faiss HNSW 索引。


if __name__ == "__main__":
    main()