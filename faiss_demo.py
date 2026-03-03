import numpy as np
import faiss

# 使用 Faiss 构建 HNSW 索引并做相似度搜索（L2 距离）
def hnsw_example():
    """
    使用 Faiss 构建 HNSW 索引并做相似度搜索（L2 距离）。
    """
    # 向量维度
    d = 64          # 每个向量的维度
    nb = 1000       # 索引中的向量数量
    nq = 5          # 查询向量数量
    k = 4           # 返回前 k 个相似向量
    M = 32          # HNSW 图中每个点的最大邻居数

    # 1. 生成一些随机向量（模拟你的向量数据）
    np.random.seed(123)
    xb = np.random.random((nb, d)).astype('float32')   # 数据库向量
    xq = np.random.random((nq, d)).astype('float32')   # 查询向量

    # 2. 创建 HNSW 索引（基于 L2 距离的扁平向量）
    index = faiss.IndexHNSWFlat(d, M)

    # 3. 调参（速度 / 精度权衡）
    index.hnsw.efConstruction = 200  # 建图时搜索宽度，越大建索引越慢、越准
    index.hnsw.efSearch = 64         # 查询时搜索宽度，越大查询越慢、越准

    # 4. 向索引中添加向量
    index.add(xb)
    print("索引中向量数量:", index.ntotal)

    # 5. 查询：给定 xq，返回距离和下标
    D, I = index.search(xq, k)   # D: 距离 (nq, k)，I: 索引 (nq, k)

    print("查询结果索引 I：")
    print(I)
    print("对应的 L2 距离 D：")
    print(D)


if __name__ == "__main__":
    hnsw_example()