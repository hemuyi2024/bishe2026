"""
DBoW3 / Bag-of-Words 检索实验工具模块

提供：
1. 图像路径解析（结合 scene_config、db_names 等）
2. 局部特征提取（SIFT / ORB）
3. ORB + DBoW3 词汇表（.dbow3 文件，需 pyDBoW3）
4. SIFT + 自建词汇表（KMeans，无预训练词典）
5. BoW 向量与 TF-IDF 加权（SIFT 用）
6. 检索接口（top-k 最近邻）

- ORB：使用预训练 DBoW3 词典（如 orbvoc.dbow3），需 pip/源码安装 pyDBoW3
- SIFT：无通用预训练词典，需从 DB 图像采样用 KMeans 自建
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pyDBoW3 as bow
    HAS_PYDBOW3 = True
except ImportError:
    bow = None
    HAS_PYDBOW3 = False


# ---------------------------------------------------------------------------
# 1. 图像路径解析
# ---------------------------------------------------------------------------

def resolve_image_path(
    name: str,
    scene_name: str,
    datasets_base: Path,
) -> Path:
    """
    从 db_names / query_names 中的 name 解析到磁盘上的实际路径。

    - DB: name 形如 "tif/220_1756_1906.tif" 或 "tif_city1/100_856_1006.tif"，
      实际位于 datasets_base / scene_name / tif_{scene_name} / <filename>
    - Query: name 形如 "uav_city1/001.jpg"，实际位于 datasets_base / scene_name / uav_{scene_name} / <filename>
    """
    base = Path(datasets_base)
    parts = name.replace("\\", "/").split("/")
    filename = parts[-1]
    if len(parts) > 1:
        subdir = parts[0]
        # "tif" -> tif_{scene_name}, "uav" 或 "uav_xxx" -> uav_{scene_name}, 其他如 tif_city1/uav_city1 用原名
        if subdir == "tif":
            subdir = f"tif_{scene_name}"
        elif subdir == "uav" or subdir.startswith("uav_"):
            subdir = f"uav_{scene_name}"
    else:
        subdir = f"tif_{scene_name}"
    return base / scene_name / subdir / filename


def get_all_image_paths(
    names: List[str],
    scene_ranges: List[Tuple[str, int, int]],
    datasets_base: Optional[Path] = None,
) -> List[Path]:
    """
    scene_ranges: [(scene_name, start, end), ...]
    返回与 names 一一对应的 Path 列表。
    """
    if datasets_base is None:
        from scene_config import DATASETS_BASE
        datasets_base = Path(DATASETS_BASE)
    base = Path(datasets_base)
    paths = []
    for scene_name, start, end in scene_ranges:
        for i in range(start, end):
            path = resolve_image_path(names[i], scene_name, base)
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 2. 特征提取
# ---------------------------------------------------------------------------

def extract_sift_features(img_path: Path, n_features: int = 2000):
    """
    从图像提取 SIFT 特征，返回 (keypoints, descriptors)。
    descriptors: (N, 128) float32，若无特征则 None。
    """
    if not HAS_OPENCV:
        raise RuntimeError("需要 OpenCV: pip install opencv-contrib-python")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], None
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp, desc = sift.detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return kp, None
    return kp, desc.astype(np.float32)


def extract_orb_features(img_path: Path, n_features: int = 2000):
    """
    从图像提取 ORB 特征。返回 (keypoints, descriptors)。
    descriptors: (N, 32) uint8，每行 256 位。
    注意：ORB 为二值特征，需专门词汇表（如 orbvoc.dbow3）才能发挥最佳效果。
    """
    if not HAS_OPENCV:
        raise RuntimeError("需要 OpenCV: pip install opencv-contrib-python")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], None
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, desc = orb.detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return kp, None
    return kp, desc  # (N, 32) uint8


# ---------------------------------------------------------------------------
# 2.1 ORB + DBoW3 词典（预训练 .dbow3，需 pyDBoW3）
# ---------------------------------------------------------------------------

def orb_descriptors_to_cvmat(desc: np.ndarray):
    """将 ORB 描述子 (N, 32) uint8 转为 pyDBoW3 可用的格式。"""
    if desc is None or len(desc) == 0:
        return None
    # OpenCV/pyDBoW3 通常接受 numpy (N, 32) uint8 或 cv2.Mat
    return np.ascontiguousarray(desc, dtype=np.uint8)


class DBoW3Database:
    """
    使用 pyDBoW3 + 预训练 ORB 词典（.dbow3）的检索数据库。
    需安装 pyDBoW3（通常需从源码编译）。
    """

    def __init__(self, vocab_path: str, n_orb_features: int = 2000):
        if not HAS_PYDBOW3:
            raise RuntimeError(
                "使用 ORB+DBoW3 词典需安装 pyDBoW3。"
                "该库常需从源码编译: https://github.com/foxis/pyDBoW3"
            )
        self.vocab_path = str(vocab_path)
        self.n_orb_features = n_orb_features
        self._voc = bow.Vocabulary()
        self._voc.load(self.vocab_path)
        self._db = bow.Database()
        self._db.setVocabulary(self._voc)
        self._n_entries = 0

    def add_from_path(self, img_path: Path) -> int:
        """添加一张 DB 图像的 ORB 特征，返回 entry id。"""
        _, desc = extract_orb_features(img_path, n_features=self.n_orb_features)
        return self.add_descriptors(desc)

    def add_descriptors(self, desc: Optional[np.ndarray]) -> int:
        """添加单张图的 ORB 描述子，返回 entry id。无特征时返回 -1 并添加空 entry。"""
        mat = orb_descriptors_to_cvmat(desc)
        if mat is None:
            # DBoW3 可能需要至少一个描述子，用全零占位（或跳过）
            mat = np.zeros((1, 32), dtype=np.uint8)
        self._db.add(mat)
        eid = self._n_entries
        self._n_entries += 1
        return eid

    def query_from_path(self, img_path: Path, max_results: int = 20):
        """查询单张图，返回 [(entry_id, score), ...]，按 score 降序。"""
        _, desc = extract_orb_features(img_path, n_features=self.n_orb_features)
        return self.query_descriptors(desc, max_results)

    def query_descriptors(
        self, desc: Optional[np.ndarray], max_results: int = 20
    ):
        """用单张图的 ORB 描述子查询，返回 [(entry_id, score), ...]。"""
        mat = orb_descriptors_to_cvmat(desc)
        if mat is None:
            return []
        ret = self._db.query(mat, max_results=max_results)
        # pyDBoW3 返回格式因版本而异: [(id,score),...] 或 QueryResults 对象等
        if ret is None or (hasattr(ret, "__len__") and len(ret) == 0):
            return []
        try:
            out = []
            for r in ret:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    out.append((int(r[0]), float(r[1])))
                elif hasattr(r, "Id") and hasattr(r, "Score"):
                    out.append((int(r.Id), float(r.Score)))
                else:
                    out.append((int(r[0]), float(r[1])))
            return out
        except Exception:
            return []

    def add_all_from_paths(
        self, paths: List[Path], show_progress: bool = True
    ) -> None:
        """批量添加 DB 图像。"""
        for i, p in enumerate(paths):
            if show_progress and (i + 1) % 500 == 0:
                print(f"  DBoW3 add {i+1}/{len(paths)} ...", flush=True)
            self.add_from_path(p)

    def query_batch(
        self, paths: List[Path], k: int = 20, show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量查询，返回 (D, I)。
        D: (N_q, k) 距离（越小越相似，1 - score）
        I: (N_q, k) 数据库索引
        """
        nq = len(paths)
        I = np.full((nq, k), -1, dtype=np.int64)
        D = np.full((nq, k), 1e9, dtype=np.float32)
        for i, p in enumerate(paths):
            if show_progress and (i + 1) % 500 == 0:
                print(f"  DBoW3 query {i+1}/{nq} ...", flush=True)
            results = self.query_from_path(p, max_results=k)
            for j, (eid, score) in enumerate(results):
                if j >= k:
                    break
                I[i, j] = int(eid)
                # DBoW3 score 通常越大越相似，转为距离
                D[i, j] = 1.0 - float(score)
        return D, I


# ---------------------------------------------------------------------------
# 3. 词汇表构建（基于 float 描述子，适用于 SIFT）
# ---------------------------------------------------------------------------

def build_vocabulary(
    descriptor_list: List[np.ndarray],
    vocab_size: int = 1024,
    sample_per_image: int = 100,
) -> "MiniBatchKMeans":
    """
    从多张图的描述子中采样，用 MiniBatchKMeans 构建词汇表。
    descriptor_list: 每元素为 (N_i, D) 的数组，可为 None（跳过）。
    返回拟合好的 KMeans 模型（用于 predict）。
    """
    if not HAS_SKLEARN:
        raise RuntimeError("需要 sklearn: pip install scikit-learn")
    all_desc = []
    for desc in descriptor_list:
        if desc is None or len(desc) == 0:
            continue
        n = min(sample_per_image, len(desc))
        idx = np.random.choice(len(desc), n, replace=(n > len(desc)))
        all_desc.append(desc[idx])
    if not all_desc:
        raise ValueError("没有有效的描述子可用于构建词汇表")
    X = np.vstack(all_desc).astype(np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        random_state=42,
        batch_size=1000,
        max_iter=100,
    )
    kmeans.fit(X)
    return kmeans


# ---------------------------------------------------------------------------
# 4. BoW 向量与 TF-IDF
# ---------------------------------------------------------------------------

def compute_bow_histogram(
    descriptors: Optional[np.ndarray],
    vocabulary: "MiniBatchKMeans",
) -> np.ndarray:
    """
    将一张图的描述子量化为 BoW 直方图 (vocab_size,)。
    若 descriptors 为空则返回全零。
    """
    vocab_size = vocabulary.n_clusters
    hist = np.zeros(vocab_size, dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return hist
    labels = vocabulary.predict(descriptors.astype(np.float32))
    for l in labels:
        hist[l] += 1.0
    return hist


def compute_tfidf_vectors(
    histograms: np.ndarray,
) -> np.ndarray:
    """
    histograms: (N_images, vocab_size)
    返回 TF-IDF 加权后的 (N_images, vocab_size)，已 L2 归一化。
    """
    N, V = histograms.shape
    # tf: 词频（每行归一化）
    tf = histograms + 1e-9
    tf = tf / tf.sum(axis=1, keepdims=True)
    # idf: 逆文档频率
    df = (histograms > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1
    tfidf = tf * idf
    normalize(tfidf, norm="l2", axis=1, copy=False)
    return tfidf.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. BoW 检索数据库
# ---------------------------------------------------------------------------

class BoWDatabase:
    """
    BoW 检索数据库：存储 DB 的 BoW 向量，支持批量 top-k 查询。
    相似度：内积（因已 L2 归一化，等价于余弦相似度）。
    """

    def __init__(self, db_vectors: np.ndarray):
        """
        db_vectors: (N_db, vocab_size) float32，已 TF-IDF 加权并 L2 归一化。
        """
        self.db_vectors = db_vectors.astype(np.float32)

    def query(self, query_vectors: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vectors: (N_q, vocab_size)
        返回 (D, I): 距离 (N_q, k) 与 索引 (N_q, k)。
        距离 = -内积（越大越相似，取负后 Faiss/MinBatch 可用「最小」找最近邻）
        为兼容习惯，这里返回的 D 实际是 1 - cosine_sim，即越小越相似。
        """
        # 内积 = 相似度（已归一化）
        scores = query_vectors @ self.db_vectors.T  # (N_q, N_db)
        # 取 top-k 最大内积
        topk = min(k, scores.shape[1])
        idx = np.argsort(-scores, axis=1)[:, :topk]  # (N_q, k)
        top_scores = np.take_along_axis(scores, idx, axis=1)
        # 返回「距离」：1 - cosine_sim，便于与 Faiss 接口一致（越小越近）
        D = 1.0 - top_scores
        return D, idx
