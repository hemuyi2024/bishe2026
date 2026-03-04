"""
在线对数域 HMM，用于在 HNSW Top-K 候选上做时序平滑。
支持无坐标模式（database_coords=None 时使用均匀转移）。
"""

import numpy as np


# 默认全局参数（可被 OnlineHMM 构造参数覆盖）
DEFAULT_UAV_SPEED = 15.0
DEFAULT_DELTA_T = 1.0
DEFAULT_ALPHA_SIGMA = 0.5
DEFAULT_EPSILON = 1e-12


class OnlineHMM:
    def __init__(
        self,
        database_coords,
        K,
        uav_speed=DEFAULT_UAV_SPEED,
        delta_t=DEFAULT_DELTA_T,
        alpha_sigma=DEFAULT_ALPHA_SIGMA,
        epsilon=DEFAULT_EPSILON,
    ):
        """
        database_coords: (N_db, 2) 每个 DB 子图的 2D 坐标，或 None 表示无坐标模式（均匀转移）
        K: 每帧候选数量
        """
        self.coords = database_coords
        self.K = K
        self.uav_speed = uav_speed
        self.delta_t = delta_t
        self.motion_radius = uav_speed * delta_t
        self.alpha_sigma = alpha_sigma
        self.epsilon = epsilon

        self.prev_log_delta = None
        self.prev_candidates = None
        self.prev_best_global = None

    def emission_log_prob(self, distances):
        """distances: 越小越相似（如 L2 距离或 1-IP）"""
        scaled = -20.0 * distances
        log_probs = scaled - np.log(np.sum(np.exp(scaled)))
        return log_probs

    def compute_transition_log(self, prev_indices, curr_indices):
        M = len(prev_indices)
        N = len(curr_indices)
        log_trans = np.full((M, N), -np.inf)

        if self.coords is None:
            # 无坐标模式：均匀转移
            log_trans[:, :] = np.log(self.epsilon + 1.0 / max(N, 1))
            return log_trans

        # 坐标含 nan 时也退化为均匀
        prev_coords = self.coords[prev_indices]
        curr_coords = self.coords[curr_indices]
        if np.any(np.isnan(prev_coords)) or np.any(np.isnan(curr_coords)):
            log_trans[:, :] = np.log(self.epsilon + 1.0 / max(N, 1))
            return log_trans

        dists = np.linalg.norm(
            prev_coords[:, None, :] - curr_coords[None, :, :],
            axis=2,
        )

        median_dist = np.median(dists)
        sigma = self.alpha_sigma * median_dist + 1e-6

        for i in range(M):
            for j in range(N):
                if self.prev_best_global is not None:
                    center = self.coords[self.prev_best_global]
                    if np.any(np.isnan(center)) or np.linalg.norm(curr_coords[j] - center) > 2 * self.motion_radius:
                        continue

                d = dists[i, j]
                diff = d - self.motion_radius
                prob = np.exp(-(diff ** 2) / (2 * sigma ** 2)) + self.epsilon
                log_trans[i, j] = np.log(prob)

        return log_trans

    def update(self, candidate_indices, distances):
        """
        candidate_indices: (K,) 当前帧 Top-K 的 DB 索引
        distances: (K,) 当前帧与候选的距离，越小越相似
        返回: 本帧最优的全局 DB 索引
        """
        log_emission = self.emission_log_prob(distances)

        if self.prev_log_delta is None:
            self.prev_log_delta = log_emission
            self.prev_candidates = candidate_indices
            best_idx = candidate_indices[np.argmax(log_emission)]
            self.prev_best_global = best_idx
            return best_idx

        log_trans = self.compute_transition_log(
            self.prev_candidates,
            candidate_indices,
        )

        curr_log_delta = np.full(self.K, -np.inf)
        for j in range(self.K):
            values = self.prev_log_delta + log_trans[:, j]
            curr_log_delta[j] = np.max(values) + log_emission[j]

        self.prev_log_delta = curr_log_delta
        self.prev_candidates = candidate_indices
        best_local = np.argmax(curr_log_delta)
        best_global = candidate_indices[best_local]
        self.prev_best_global = best_global
        return best_global


# =========================
# Demo：随机数据模拟在线运行
# =========================

def _run_demo():
    import faiss
    np.random.seed(0)
    N, D, T, K = 500, 128, 30, 10
    database_features = np.random.randn(N, D).astype("float32")
    database_features /= np.linalg.norm(database_features, axis=1, keepdims=True)
    database_coords = np.random.uniform(0, 1000, (N, 2))

    M = 16
    index = faiss.IndexHNSWFlat(D, M)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50
    index.add(database_features)

    hmm = OnlineHMM(database_coords, K)
    print("Faiss + Online HMM Demo:\n")
    for t in range(T):
        query = np.random.randn(D).astype("float32")
        query /= np.linalg.norm(query)
        distances, labels = index.search(query.reshape(1, -1), K)
        best = hmm.update(labels[0], distances[0])
        print(f"Frame {t:02d} -> Best Submap Index: {best}")


if __name__ == "__main__":
    _run_demo()
