from pathlib import Path

import faiss
import numpy as np

from demo_readH5 import (
    load_netvlad_descriptors,
    load_multi_netvlad_descriptors,
    DB_H5_PATH,
    DB_H5_PATHS,
    QUERY_H5_PATH,
)


def build_hnsw_index(
    vectors: np.ndarray,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> faiss.IndexHNSWFlat:
    """
    使用 Faiss 构建 HNSW 索引（L2 距离）。
    vectors: 形状为 (N, D) 的 float32 数组。
    """
    d = vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vectors)
    return index


def get_or_build_index(
    db_vectors: np.ndarray,
    index_path: Path,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> faiss.IndexHNSWFlat:
    """
    如果本地已有 HNSW 索引文件，则直接加载；
    否则重新构建并保存到本地。
    """
    if index_path.exists():
        print(f"检测到已有索引文件，直接加载: {index_path}")
        index = faiss.read_index(str(index_path))
        # 运行时查询时使用的 efSearch 仍然可以调整
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = ef_search
        return index

    print("未检测到索引文件，开始构建 HNSW 索引...")
    index = build_hnsw_index(
        db_vectors,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"索引已构建并保存到: {index_path}")
    return index


def save_retrieval_results_txt(
    out_path: Path,
    query_names,
    db_names,
    indices: np.ndarray,
    distances: np.ndarray,
):
    """
    将检索结果保存到 txt 文件中。
    每一行格式：query_name db_name distance
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        num_queries, topk = indices.shape
        for qi in range(num_queries):
            q_name = query_names[qi]
            for rank in range(topk):
                db_idx = indices[qi, rank]
                dist = distances[qi, rank]
                db_name = db_names[db_idx]
                f.write(f"{q_name} {db_name} {dist:.6f}\n")


def main():
    # 1. 读取数据库和查询的 NetVLAD 描述子
    print("加载数据库 NetVLAD 描述子（多 DB 文件合并）...")
    db_names, db_descs = load_multi_netvlad_descriptors(DB_H5_PATHS)
    print(f"DB: {len(db_names)} 条，特征维度 {db_descs.shape[1]}")

    print("加载查询 NetVLAD 描述子（单一 QUERY_H5_PATH）...")
    query_names, query_descs = load_netvlad_descriptors(QUERY_H5_PATH)
    print(f"Query: {len(query_names)} 条，特征维度 {query_descs.shape[1]}")

    # 2. 构建或加载 HNSW 索引
    # 输出路径可按需修改，这里放在 RealUAV/hnsw_multi_db 目录下
    output_root = Path("/home/lty/outputs/RealUAV/hnsw_multi_db")
    index_path = output_root / "netvlad-hnsw.index"
    index = get_or_build_index(
        db_descs,
        index_path=index_path,
        M=32,
        ef_construction=200,
        ef_search=64,
    )
    print("索引中向量数量:", index.ntotal)

    # 3. 检索（top-5 结果）
    topk = 5
    print(f"对每个查询向量检索 top-{topk} 相似数据库图像...")
    D, I = index.search(query_descs, topk)

    # 4. 保存 top-5 结果到 txt
    out_txt = output_root / "pairs-query-netvlad-hnsw-top5.txt"
    save_retrieval_results_txt(out_txt, query_names, db_names, I, D)

    print(f"Top-{topk} 检索结果已保存到: {out_txt}")


if __name__ == "__main__":
    main()