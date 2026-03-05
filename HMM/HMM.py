"""
在线对数域 HMM，用于在 HNSW Top-K 候选上做时序平滑。
支持无坐标模式（database_coords=None 时使用均匀转移）。
"""

import numpy as np


# 默认全局参数（可被 OnlineHMM 构造参数覆盖）
DEFAULT_UAV_SPEED = 20.1 #pixel/s
DEFAULT_DELTA_T = 1/3.0
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
        verbose=False,
    ):
        """
        database_coords: (N_db, 2) 每个 DB 子图的 2D 坐标，或 None 表示无坐标模式（均匀转移）
        K: 每帧候选数量
        verbose: 是否打印每帧关键参数与中间结果（用于调试第一个场景）
        """
        self.coords = database_coords
        self.K = K
        self.uav_speed = uav_speed
        self.delta_t = delta_t
        self.motion_radius = uav_speed * delta_t
        self.alpha_sigma = alpha_sigma
        self.epsilon = epsilon
        self.verbose = bool(verbose)

        self.prev_log_delta = None
        self.prev_candidates = None
        self.prev_best_global = None
        self._call_count = 0

    def emission_log_prob(self, distances):
        """distances: 越小越相似（如 L2 距离或 1-IP）"""
        scaled = -20.0 * distances
        log_probs = scaled - np.log(np.sum(np.exp(scaled)))
        return log_probs

    def compute_transition_log(self, prev_indices, curr_indices, displacement=None):
        """
        displacement: (dx, dy) 可选，与 coords 同单位的位移；若提供则用 center = 上一帧中心 + displacement 作为期望位置。
        """
        M = len(prev_indices)
        N = len(curr_indices)
        log_trans = np.full((M, N), -np.inf)

        if self.coords is None:
            log_trans[:, :] = np.log(self.epsilon + 1.0 / max(N, 1))
            return log_trans

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

        if self.prev_best_global is not None:
            center = np.array(self.coords[self.prev_best_global], dtype=np.float64)
            radius = 2 * self.motion_radius
            if displacement is not None:
                dx, dy = float(displacement[0]), float(displacement[1])
                center = center + np.array([dx, dy])
                d_norm = np.sqrt(dx * dx + dy * dy)
                if d_norm > 1e-9:
                    radius = 2 * max(self.motion_radius, d_norm)
        else:
            center = None
            radius = 2 * self.motion_radius

        if self.verbose:
            print(f"    [transition] prev_best_global={self.prev_best_global}, center={center}, radius={radius:.1f}, sigma={sigma:.4f}, displacement={displacement}")

        for i in range(M):
            for j in range(N):
                if center is not None:
                    if np.any(np.isnan(center)) or np.linalg.norm(curr_coords[j] - center) > radius:
                        continue
                if displacement is not None and center is not None:
                    dist_to_center = np.linalg.norm(curr_coords[j] - center)
                    prob = np.exp(-(dist_to_center ** 2) / (2 * sigma ** 2)) + self.epsilon
                else:
                    d = dists[i, j]
                    diff = d - self.motion_radius
                    prob = np.exp(-(diff ** 2) / (2 * sigma ** 2)) + self.epsilon
                log_trans[i, j] = np.log(prob)

        if self.verbose:
            valid = np.isfinite(log_trans)
            n_valid = int(np.sum(valid))
            if n_valid > 0:
                vals = log_trans[valid]
                print(f"    [transition] log_trans: shape={log_trans.shape}, 有效元素={n_valid}, max={np.max(vals):.4f}, min={np.min(vals):.4f}")
            else:
                print(f"    [transition] log_trans: 全为 -inf")

        return log_trans

    def override_prev_best(self, global_idx: int):
        """将上一帧最优设为 global_idx，下一帧转移将以此位置为中心（用于 warm-start 等）。"""
        self.prev_best_global = global_idx

    def update(self, candidate_indices, distances, return_top_k=10, displacement=None):
        """
        candidate_indices: (K,) 当前帧 Top-K 的 DB 索引
        distances: (K,) 当前帧与候选的距离，越小越相似
        return_top_k: 返回按后验概率排序的前几个索引，默认 10
        displacement: (dx, dy) 可选，与 coords 同单位的帧间位移，用于更准的转移先验
        返回: list of int，长度 min(return_top_k, K)，按后验概率从高到低排序
        """
        frame_id = self._call_count
        self._call_count += 1

        log_emission = self.emission_log_prob(distances)
        K_curr = len(candidate_indices)
        top_k = min(int(return_top_k), K_curr)

        if self.verbose:
            print(f"\n--- HMM Frame {frame_id} ---")
            print(f"  displacement = {displacement}")
            print(f"  candidate_indices (HNSW top-K) = {candidate_indices}")
            print(f"  distances (越小越相似) = {distances}")
            print(f"  log_emission: min={log_emission.min():.4f}, max={log_emission.max():.4f}, argmax={np.argmax(log_emission)} -> global_idx={candidate_indices[np.argmax(log_emission)]}")

        if self.prev_log_delta is None:
            self.prev_log_delta = log_emission
            self.prev_candidates = candidate_indices
            order = np.argsort(-log_emission)
            ranked = candidate_indices[order]
            self.prev_best_global = int(ranked[0])
            if self.verbose:
                print(f"  [首帧] 无转移，按 emission 排序 -> top-3 global_idx = {[int(ranked[j]) for j in range(min(3, top_k))]}")
            return [int(ranked[j]) for j in range(top_k)]

        if self.verbose:
            print(f"  prev_best_global = {self.prev_best_global}, prev_candidates = {self.prev_candidates}")
            if self.coords is not None and self.prev_best_global is not None:
                c = self.coords[self.prev_best_global]
                print(f"  prev_best 坐标 (大图像素) = ({c[0]:.1f}, {c[1]:.1f})")

        log_trans = self.compute_transition_log(
            self.prev_candidates,
            candidate_indices,
            displacement=displacement,
        )

        curr_log_delta = np.full(self.K, -np.inf)
        for j in range(self.K):
            values = self.prev_log_delta + log_trans[:, j]
            curr_log_delta[j] = np.max(values) + log_emission[j]

        self.prev_log_delta = curr_log_delta.copy()
        self.prev_candidates = candidate_indices
        order = np.argsort(-curr_log_delta)
        ranked = candidate_indices[order]
        self.prev_best_global = int(ranked[0])

        if self.verbose:
            top3_j = order[:3]
            print(f"  curr_log_delta (后验) top-3: j={top3_j}, global_idx={[int(candidate_indices[j]) for j in top3_j]}, log_delta={[curr_log_delta[j] for j in top3_j]}")
            print(f"  HMM 输出 top-1 = {self.prev_best_global}, top-3 = {[int(ranked[j]) for j in range(min(3, top_k))]}")

        return [int(ranked[j]) for j in range(top_k)]


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
        top = hmm.update(labels[0], distances[0], return_top_k=5)
        print(f"Frame {t:02d} -> Top-1: {top[0]}, Top-5: {top}")


if __name__ == "__main__":
    _run_demo()
