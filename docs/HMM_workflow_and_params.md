# HMM 工作流程与参数/变量总结

基于 `run_hnsw_hmm.ipynb`、`HMM/HMM.py`、`coords_gt_utils.py`、`scene_config.py` 的代码梳理。

---

## 一、整体数据流（与 HMM 的关系）

1. **场景配置**（`scene_config.py`）  
   - 从 H5 路径解析场景名，得到每条轨迹的 `(scene_name, start, end)` 以及每个场景的 `mapbox_path`、`tif_dir`、`uav_infos_path`。

2. **DB 坐标与 GT**（`coords_gt_utils.py`）  
   - `build_database_coords(db_names, db_scene_ranges, ...)` → `database_coords`：形状 `(N_db, 2)`，每个 DB 子图在大图中的中心像素坐标（大图 EPSG:3857）；缺坐标处为 `nan`。  
   - `build_gt_9_for_all_trajectories(...)` → `gt_9_list`（每 query 的 25 张真值集合）、`gt_25_ordered`（每 query 的 25 张真值按到 query 距离排序的列表），用于 Recall@25 评估。

3. **HNSW 检索**  
   - 每条轨迹在其对应场景的 DB 子集上做 HNSW 检索，得到每帧的 Top-K 索引 `I_t` 和距离 `D_t`。  
   - 给 HMM 的“距离”为 **越小越相似**：`dist_for_hmm = 1.0 - D_t`（因 Faiss 内积越大越相似）。

4. **Query 像素坐标（用于 displacement）**  
   - 用 `uav_infos.csv` 的经纬度 + 场景的 `mapbox` geotransform，经 `lonlat_to_pixel` 得到每帧在大图中的像素坐标 `query_px[f]`, `query_py[f]`。  
   - 帧间位移：`displacement = (query_px[f] - query_px[f-1], query_py[f] - query_py[f-1])`，单位与大图一致（像素）。

5. **HMM**  
   - 输入：当前帧的 Top-K 候选索引 `inds`、对应的“距离”`dists`、可选的 `displacement`。  
   - 输出：按后验概率排序的 Top-`return_top_k` 的 DB 索引（全局索引），用于 Recall 与保存结果。

---

## 二、HMM 工作流程（`HMM/HMM.py` 中的 `OnlineHMM`）

### 2.1 设计要点

- **在线、对数域**：每帧只依赖上一帧的 log-delta，不回溯整条轨迹。  
- **状态空间**：每帧的状态 = 当前帧的 K 个候选（DB 索引），即状态集随帧变化。  
- **发射**：由 HNSW 给出的距离转为 log 发射概率（距离越小概率越大）。  
- **转移**：由 DB 坐标 + 可选位移得到“期望下一帧位置”，用高斯形式的距离/位移惩罚；无坐标时退化为均匀转移。

### 2.2 单帧 `update` 流程

对每一帧调用一次 `hmm.update(candidate_indices, distances, return_top_k=10, displacement=None)`：

| 步骤 | 说明 |
|------|------|
| 1 | **发射概率（对数）**：`log_emission = emission_log_prob(distances)`，距离越小 log_emission 越大。 |
| 2 | **首帧**：若 `prev_log_delta is None`，则无转移；`prev_log_delta = log_emission`，`prev_candidates = candidate_indices`，按 emission 排序得到本帧输出，并设 `prev_best_global = 排序后第 0 个全局索引`，然后返回。 |
| 3 | **非首帧**：调用 `compute_transition_log(prev_candidates, candidate_indices, displacement)` 得到 `log_trans`（形状 M×N，M=上帧候选数，N=本帧候选数）。 |
| 4 | **前向递推**：对每个本帧候选 j，`curr_log_delta[j] = max_i(prev_log_delta[i] + log_trans[i,j]) + log_emission[j]`（即 Viterbi 式的一步 max-product）。 |
| 5 | **更新状态**：`prev_log_delta = curr_log_delta`，`prev_candidates = candidate_indices`，`prev_best_global = 按 curr_log_delta 排序后的第 0 个候选对应的全局索引`。 |
| 6 | **返回**：按 `curr_log_delta` 从大到小排序的 `return_top_k` 个全局 DB 索引。 |

### 2.3 发射概率

- **输入**：`distances`，长度 K，**越小表示越相似**（如 L2 或 `1 - 内积`）。  
- **计算**：`scaled = -20.0 * distances`，再对 `exp(scaled)` 做 log-softmax 得到 `log_emission`。  
- **含义**：把“距离”转成在 K 个候选上的归一化对数发射概率，距离越小权重越大（系数 20 控制锐度）。

### 2.4 转移概率（`compute_transition_log`）

- **无坐标模式**（`database_coords is None`）：  
  - `log_trans[i,j] = log(epsilon + 1/N)`，即均匀转移。

- **有坐标模式**：  
  - 用 `prev_candidates` 和当前 `candidate_indices` 对应的坐标算距离矩阵 `dists`（上一帧 M 个点到当前帧 N 个点的欧氏距离）。  
  - **sigma**：`median_dist = median(dists)`，`sigma = alpha_sigma * median_dist + 1e-6`。  
  - **中心与半径**：  
    - 若有 `prev_best_global`：`center = coords[prev_best_global]`；若提供 `displacement=(dx,dy)`，则 `center += (dx, dy)`，且 `radius = 2 * max(motion_radius, ||displacement||)`，否则 `radius = 2 * motion_radius`。  
    - `motion_radius = uav_speed * delta_t`（一帧内最大运动距离，像素）。  
  - 对每个 (i,j)：若当前帧候选 j 的坐标在 `center` 的 `radius` 外，则 `log_trans[i,j] = -inf`；否则：  
    - 若用了 displacement：`prob = exp(-dist_to_center^2 / (2*sigma^2)) + epsilon`（到“期望中心”的高斯）；  
    - 否则：`d = dists[i,j]`，`prob = exp(-(d - motion_radius)^2 / (2*sigma^2)) + epsilon`。  
  - `log_trans[i,j] = log(prob)`。

---

## 三、参数汇总

### 3.1 `OnlineHMM` 构造参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `database_coords` | `(N_db, 2)` 或 `None` | 必填 | 每个 DB 子图在大图中的 2D 中心坐标；`None` 表示无坐标（均匀转移）。 |
| `K` | int | 必填 | 每帧候选数量，与 HNSW 的 Top-K 一致。 |
| `uav_speed` | float | `20.1` | 假设的 UAV 速度（像素/秒），用于 `motion_radius = uav_speed * delta_t`。 |
| `delta_t` | float | `1/3.0` | 相邻两帧时间间隔（秒）。 |
| `alpha_sigma` | float | `0.5` | 转移高斯尺度：`sigma = alpha_sigma * median_dist + 1e-6`。 |
| `epsilon` | float | `1e-12` | 转移概率及均匀转移的下界，避免 log(0)。 |
| `verbose` | bool | `False` | 是否打印每帧的 displacement、candidates、log_emission、transition、curr_log_delta 等。 |

### 3.2 `update` 参数

| 参数 | 类型 | 含义 |
|------|------|------|
| `candidate_indices` | `(K,)` int 数组 | 当前帧 HNSW Top-K 的 **全局** DB 索引。 |
| `distances` | `(K,)` float 数组 | 与候选对应的距离，**越小越相似**（如 `1 - 内积`）。 |
| `return_top_k` | int | 返回按后验排序的前几个索引，默认 10。 |
| `displacement` | `(dx, dy)` 或 `None` | 与 coords 同单位的帧间位移；有则转移中心为上一帧最优位置 + displacement。 |

### 3.3 全局默认常量（`HMM.py`）

| 常量 | 值 | 含义 |
|------|-----|------|
| `DEFAULT_UAV_SPEED` | 20.1 | 像素/秒 |
| `DEFAULT_DELTA_T` | 1/3.0 | 秒 |
| `DEFAULT_ALPHA_SIGMA` | 0.5 | 转移 sigma 系数 |
| `DEFAULT_EPSILON` | 1e-12 | 概率下界 |

### 3.4 外部传入的变量（notebook/run_hnsw_hmm.py）

| 变量 | 来源 | 用途 |
|------|------|------|
| `database_coords` | `coords_gt_utils.build_database_coords(...)` | 传入 `OnlineHMM(..., database_coords=coords_for_hmm, ...)`；`coords_for_hmm = database_coords if use_coords else None`。 |
| `K` | HNSW 的 K（如 20） | 与每帧候选数一致。 |
| `query_px`, `query_py` | `lonlat_to_pixel(lons[f], lats[f], gt_tuple)` 按帧计算 | 得到 `displacement = (query_px[f]-query_px[f-1], query_py[f]-query_py[f-1])`。 |
| `inds`, `dists` | `I_t[f]`（全局索引）、`dist_for_hmm[f]`（`1.0 - D_t[f]`） | 每帧传入 `hmm.update(inds, dists, return_top_k=10, displacement=displacement)`。 |

---

## 四、HMM 内部状态变量（每帧更新）

| 变量 | 形状/类型 | 含义 |
|------|-----------|------|
| `prev_log_delta` | `(K,)` float | 上一帧的 log-delta（各候选的对数后验），首帧前为 `None`。 |
| `prev_candidates` | `(K,)` int | 上一帧的 K 个候选的**全局** DB 索引。 |
| `prev_best_global` | int | 上一帧后验最大的候选的全局 DB 索引；用于本帧转移的“中心”和 radius。 |
| `_call_count` | int | 已调用 `update` 的次数（帧号）。 |

### 其他方法

- **`override_prev_best(global_idx)`**：将 `prev_best_global` 设为 `global_idx`，下一帧的转移中心会基于该位置（用于 warm-start：用 GT top1 替代预测并让 HMM 从该位置继续）。

---

## 五、Notebook/脚本中的 HMM 调用与 warm-start

- 每条轨迹一个 `OnlineHMM` 实例；`coords_for_hmm` 为全库 `database_coords` 或 `None`（由 `use_coords = (n_valid > 0)` 决定）。  
- 对每一帧 f：  
  - 取 `inds = I_t[f]`，`dists = dist_for_hmm[f]`；若长度不足 K 则 padding 或截断到 K。  
  - 若 f≥1 且当前帧和上一帧的 query 像素坐标存在，则 `displacement = (query_px[f]-query_px[f-1], query_py[f]-query_py[f-1])`，否则 `displacement = None`。  
  - `hmm_top = hmm.update(inds, dists, return_top_k=10, displacement=displacement)`。  
  - 若开启 warm-start 且 f < 3 且该 query 有 GT，则用 GT 的 top1 覆盖 `pred_hmm`/`topk_hmm` 的第 0 位，并调用 `hmm.override_prev_best(gt_idx)`，使下一帧转移以 GT 位置为中心。

---

## 六、与坐标/GT 相关的常量（coords_gt_utils / scene_config）

| 名称 | 值/含义 |
|------|---------|
| `MAP_EPSG` | 3857（Web Mercator） |
| `TILE_GRID_STEP` | 150（子图在大图中的网格步长，像素） |
| `DATASETS_BASE` | 数据集根目录（如 `/home/lty/datasets/RealUAV`） |
| DB 子图命名 | `id_startx_starty.tif`，解析得 (id, startx, starty)，中心为 (startx+tile_w/2, starty+tile_h/2) |
| GT@25 | 每 query 对应 uav_infos 该帧经纬度 → 大图像素 → 距离最近的 25 张子图（中心第一，其余按到 query 距离排序） |

以上即 HMM 的工作流程以及所涉及的参数与变量的完整总结。
