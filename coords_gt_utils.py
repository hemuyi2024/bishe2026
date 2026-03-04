"""
DB 子图坐标生成（GDAL mapbox + 解析 id_startx_starty）、
uav_infos.csv 读取（首行为列名）、坐标法 Recall@25 真值生成（每 query 方圆 25 张）。
"""

from pathlib import Path
from typing import List, Tuple, Optional, Set
import csv
import numpy as np

try:
    from osgeo import gdal, osr
    gdal.DontUseExceptions()  # 避免 PROJ 报错刷屏与 FutureWarning
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


# ------------------------- 解析 DB 子图命名 id_startx_starty -------------------------

def parse_db_name(name: str) -> Optional[Tuple[int, int, int]]:
    """
    从 db 图像键解析 id_startx_starty。
    支持 "10_3000_1500.tif" 或 "xxx/10_3000_1500.tif"。
    返回 (id, startx, starty) 或 None。
    """
    base = name.replace("\\", "/").split("/")[-1]
    base = base.replace(".tif", "").strip()
    parts = base.split("_")
    if len(parts) < 3:
        return None
    try:
        id_ = int(parts[0])
        startx = int(parts[1])
        starty = int(parts[2])
        return (id_, startx, starty)
    except (ValueError, IndexError):
        return None


# ------------------------- uav_infos.csv（首行为列名）-------------------------

def load_uav_infos(
    csv_path: Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 uav_infos.csv，第一行为列名。
    返回 (latitudes, longitudes)，形状均为 (n_frames,)。
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"uav_infos not found: {path}")

    lats, lons = [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and lat_col not in reader.fieldnames:
            # 尝试用第3、4列位置
            for row in reader:
                vals = list(row.values()) if isinstance(row, dict) else row
                if len(vals) >= 4:
                    try:
                        lats.append(float(vals[2]))
                        lons.append(float(vals[3]))
                    except (ValueError, IndexError):
                        pass
                elif isinstance(row, dict) and lat_col in row:
                    lats.append(float(row[lat_col]))
                    lons.append(float(row[lon_col]))
        else:
            for row in reader:
                lats.append(float(row[lat_col]))
                lons.append(float(row[lon_col]))

    lats = np.array(lats, dtype=np.float64)
    lons = np.array(lons, dtype=np.float64)
    # 若文件中经/纬列写反（如第三列实际是经度、第四列是纬度），则交换
    if lats.size > 0 and (np.any(lats > 90) or np.any(lats < -90)):
        lats, lons = lons.copy(), lats.copy()
    return lats, lons


# ------------------------- GDAL：geotransform、像素/地理转换 -------------------------
# 大图（mapbox.tif）固定为 EPSG:3857（Web Mercator），转换与显示前后一致
MAP_EPSG = 3857
# 子图左上角在大图中的网格步长（像素）；startx/starty 为像素坐标，相邻子图间隔 150 像素
TILE_GRID_STEP = 150


def _open_ds(path: Path):
    if not HAS_GDAL:
        return None
    ds = gdal.Open(str(path))
    return ds


def get_geotransform_and_srs(mapbox_path: Path):
    """返回 (geotransform tuple, wkt_or_None)。"""
    if not HAS_GDAL:
        return None, None
    ds = _open_ds(Path(mapbox_path))
    if ds is None:
        return None, None
    gt = ds.GetGeoTransform()
    srs = ds.GetProjection()
    ds = None
    return gt, srs


def get_tile_size_from_tif(tif_dir: Path) -> Optional[Tuple[int, int]]:
    """从场景 tif 目录中任选一张图读取宽高，作为统一 tile_w, tile_h。"""
    if not HAS_GDAL:
        return None
    tif_dir = Path(tif_dir)
    if not tif_dir.is_dir():
        return None
    for f in list(tif_dir.glob("*.tif"))[:5] + list(tif_dir.glob("*.tiff"))[:5]:
        ds = _open_ds(f)
        if ds is not None:
            w, h = ds.RasterXSize, ds.RasterYSize
            ds = None
            return (w, h)
    return None


def _normalize_lonlat(lon: float, lat: float) -> Tuple[float, float]:
    """
    若疑似 CSV 列颠倒（经度写进纬度列），则交换为 (lon, lat)。
    约定：经度约 [-180,180]，纬度约 [-90,90]。
    """
    lon, lat = float(lon), float(lat)
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return (lon, lat)
    if -90 <= lon <= 90 and -180 <= lat <= 180:
        return (lat, lon)  # 列颠倒：当前 lon 实为 lat，当前 lat 实为 lon
    return (lon, lat)


def _wgs84_to_web_mercator(lon: float, lat: float) -> Tuple[float, float]:
    """
    WGS84 (EPSG:4326) 经纬度 -> Web Mercator (EPSG:3857) 投影坐标（米）。
    不依赖 GDAL，用于 lonlat_to_pixel 在 GDAL 不可用或失败时的回退。
    """
    import math
    lon, lat = float(lon), float(lat)
    # 纬度限制在 Web Mercator 有效范围，避免 tan 爆炸
    lat = max(-85.051129, min(85.051129, lat))
    r = 6378137.0
    x = r * math.radians(lon)
    y = r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return (x, y)


def lonlat_to_pixel(
    lon: float,
    lat: float,
    gt: Tuple[float, ...],
    srs_wkt: Optional[str] = None,
    return_geo: bool = False,
):
    """
    将经纬度 (lon, lat) 转为大图像素坐标。
    大图固定为 EPSG:3857（MAP_EPSG），WGS84(4326) -> 3857 再逆 geotransform 得像素；
    参数 srs_wkt 仅保留兼容，内部不使用，大图 CRS 一律用 MAP_EPSG。
    return_geo=True 时返回 (px, py, x_geo, y_geo)。
    """
    lon, lat = _normalize_lonlat(lon, lat)
    x_geo, y_geo = float(lon), float(lat)
    if HAS_GDAL:
        try:
            sr_wgs = osr.SpatialReference()
            sr_wgs.ImportFromEPSG(4326)
            sr_map = osr.SpatialReference()
            sr_map.ImportFromEPSG(MAP_EPSG)
            gdal.PushErrorHandler(lambda a, b, c: None)
            try:
                ct = osr.CoordinateTransformation(sr_wgs, sr_map)
                pt = ct.TransformPoint(lon, lat)
                if pt and abs(pt[0]) < 1e30 and abs(pt[1]) < 1e30:
                    x_geo, y_geo = pt[0], pt[1]
            finally:
                gdal.PopErrorHandler()
        except Exception:
            pass

    # 若仍为经纬度（GDAL 未用或转换失败），大图为 3857 时用公式做 4326->3857
    if MAP_EPSG == 3857 and (abs(x_geo) <= 180 and abs(y_geo) <= 90):
        x_geo, y_geo = _wgs84_to_web_mercator(lon, lat)

    px = (x_geo - gt[0]) / gt[1] if abs(gt[1]) > 1e-20 else 0.0
    py = (y_geo - gt[3]) / gt[5] if abs(gt[5]) > 1e-20 else 0.0
    if return_geo:
        return (px, py, x_geo, y_geo)
    return (px, py)


# ------------------------- 单场景 DB 坐标（大图像素中心）-------------------------

def get_db_coords_for_scene(
    db_names: List[str],
    mapbox_path: Path,
    tif_dir: Path,
) -> np.ndarray:
    """
    根据 db_names 解析 id_startx_starty，结合 tile 尺寸得到每个子图在大图中的中心像素坐标。
    返回 (N, 2)，若 GDAL 不可用或路径无效则对应行为 nan。
    """
    n = len(db_names)
    coords = np.full((n, 2), np.nan, dtype=np.float64)

    tile_wh = get_tile_size_from_tif(Path(tif_dir))
    if tile_wh is None:
        return coords

    tile_w, tile_h = tile_wh
    for i, name in enumerate(db_names):
        parsed = parse_db_name(name)
        if parsed is None:
            continue
        _, startx, starty = parsed
        cx = startx + tile_w / 2.0
        cy = starty + tile_h / 2.0
        coords[i, 0], coords[i, 1] = cx, cy
    return coords


# ------------------------- 全库 database_coords（多场景拼接）-------------------------

def build_database_coords(
    db_names: List[str],
    db_scene_ranges: List[Tuple[str, int, int]],
    datasets_base: Optional[Path] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    db_scene_ranges: [(scene_name, start, end), ...]，与 db_names 顺序一致。
    返回 (len(db_names), 2)，某场景缺路径时该段填 nan。
    """
    from scene_config import get_scene_paths, DATASETS_BASE

    base = Path(datasets_base) if datasets_base is not None else DATASETS_BASE
    out = np.full((len(db_names), 2), np.nan, dtype=np.float64)
    if verbose:
        print("=== database_coords 各场景（未算出坐标的会打印原因）===")
    for scene_name, start, end in db_scene_ranges:
        n_entries = end - start
        paths = get_scene_paths(scene_name, base)
        mapbox_path = paths["mapbox_path"]
        tif_dir = paths["tif_dir"]
        map_ok = mapbox_path.exists()
        tif_ok = tif_dir.exists()
        if not map_ok or not tif_ok:
            if verbose:
                print(f"  [{scene_name}] 共 {n_entries} 条 -> 未算坐标: mapbox 存在={map_ok} ({mapbox_path}), tif_dir 存在={tif_ok} ({tif_dir})")
            continue
        seg = get_db_coords_for_scene(db_names[start:end], mapbox_path, tif_dir)
        n_valid = int(np.sum(~np.any(np.isnan(seg), axis=1)))
        out[start:end, :] = seg
        if verbose:
            print(f"  [{scene_name}] 共 {n_entries} 条 -> 有效坐标 {n_valid}/{n_entries}")
    if verbose:
        total_valid = int(np.sum(~np.any(np.isnan(out), axis=1)))
        print(f"  -> 合计有效坐标: {total_valid}/{len(db_names)}\n")
    return out


# ------------------------- 坐标法 GT：每 query 的 25 张真值（Recall@25）-------------------------
#
# 逻辑简述（便于逐行检查）：
# 1. uav_infos.csv 第 i 行 = 第 i 帧的 (latitude, longitude)，与 query 图像顺序一致。
# 2. 用 mapbox 的 geotransform 将 (lon, lat) 转成大图像素 (px, py)。
# 3. 用 tile_w/tile_h 与 DB 子图命名 id_startx_starty 建立 (startx, starty) -> 局部下标。
# 4. 找包含 (px, py) 的子图：满足 startx <= px < startx+tile_w 且 starty <= py < starty+tile_h；
#    若无则取中心距离最近的 (sx, sy)。
# 5. 该子图 + 方圆 5x5 邻域（共 25 张）在 startxy_to_local 中查到的
#    局部下标 + global_db_start = 全局 DB 索引，构成该 query 的 25 张真值集合；检索命中其中任一张即算正确。


def _build_startxy_to_local_index(db_names: List[str]) -> dict:
    """(startx, starty) -> 在 db_names 中的局部下标。"""
    out = {}
    for i, name in enumerate(db_names):
        p = parse_db_name(name)
        if p is not None:
            _, sx, sy = p
            out[(sx, sy)] = i
    return out


def _neighbor_startxy(startx: int, starty: int) -> List[Tuple[int, int]]:
    """
    中心 (startx, starty) 及方圆 5x5 邻域（共 25 个）的 (startx, starty) 列表。
    startx/starty 为子图左上角在大图中的像素坐标，步长固定为 TILE_GRID_STEP（150 像素）。
    """
    deltas = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    return [(startx + dx * TILE_GRID_STEP, starty + dy * TILE_GRID_STEP) for dx, dy in deltas]


def get_gt_9_for_trajectory(
    uav_infos_path: Path,
    scene_db_names: List[str],
    mapbox_path: Path,
    tif_dir: Path,
    global_db_start: int,
    n_query_expected: int,
    scene_name: str = "",
    debug_first_frame: bool = True,
) -> List[Set[int]]:
    """
    对一条轨迹的每一帧，根据 uav_infos 的经纬度得到「包含该点的子图 + 方圆 5x5 邻域」共 25 张真值对应的全局 DB 索引。
    约定：uav_infos 第 i 行对应该轨迹第 i 帧 query（行序与 query 顺序一致）。
    返回 list of set，长度严格为 n_query_expected；检索结果命中该集合内任一张即算正确。
    debug_first_frame=True 时会对第一帧打印：geotransform、EPSG、第一张图坐标、转大图坐标等。
    """
    lats, lons = load_uav_infos(Path(uav_infos_path))
    n_csv = len(lats)
    if n_csv != n_query_expected:
        import warnings
        warnings.warn(
            f"uav_infos 行数({n_csv})与轨迹 query 数({n_query_expected})不一致，将按 query 数截断或补空。"
        )
    n_use = min(n_csv, n_query_expected)
    if n_use == 0:
        return [set() for _ in range(n_query_expected)]

    gt_tuple, _ = get_geotransform_and_srs(Path(mapbox_path))
    if gt_tuple is None:
        return [set() for _ in range(n_query_expected)]
    epsg = f"EPSG:{MAP_EPSG}"

    tile_wh = get_tile_size_from_tif(Path(tif_dir))
    if tile_wh is None:
        return [set() for _ in range(n_query_expected)]
    tile_w, tile_h = tile_wh

    startxy_to_local = _build_startxy_to_local_index(scene_db_names)
    result = []
    x_geo, y_geo = None, None
    for i in range(n_use):
        if i == 0 and debug_first_frame:
            px, py, x_geo, y_geo = lonlat_to_pixel(
                lons[i], lats[i], gt_tuple, None, return_geo=True
            )
        else:
            px, py = lonlat_to_pixel(lons[i], lats[i], gt_tuple, None)

        best_key = None
        best_dist = float("inf")
        for (sx, sy) in startxy_to_local:
            if sx <= px < sx + tile_w and sy <= py < sy + tile_h:
                best_key = (sx, sy)
                break
            d = (px - (sx + tile_w / 2)) ** 2 + (py - (sy + tile_h / 2)) ** 2
            if d < best_dist:
                best_dist = d
                best_key = (sx, sy)
        if best_key is None:
            result.append(set())
            if i == 0 and debug_first_frame:
                _print_first_frame_debug(
                    scene_name, gt_tuple, epsg,
                    lats[i], lons[i], x_geo, y_geo, px, py,
                    tile_w, tile_h, None, set(), scene_db_names, global_db_start,
                )
            continue
        sx, sy = best_key
        neighbors = _neighbor_startxy(sx, sy)
        global_indices = set()
        for (nx, ny) in neighbors:
            local_idx = startxy_to_local.get((nx, ny))
            if local_idx is not None:
                global_indices.add(global_db_start + local_idx)
        result.append(global_indices)
        if i == 0 and debug_first_frame:
            _print_first_frame_debug(
                scene_name, gt_tuple, epsg,
                lats[i], lons[i], x_geo, y_geo, px, py,
                tile_w, tile_h, (sx, sy), global_indices, scene_db_names, global_db_start,
            )
    while len(result) < n_query_expected:
        result.append(set())
    return result[:n_query_expected]


def _print_first_frame_debug(
    scene_name: str,
    gt: Tuple[float, ...],
    epsg: str,
    lat: float,
    lon: float,
    x_geo: Optional[float],
    y_geo: Optional[float],
    px: float,
    py: float,
    tile_w: int,
    tile_h: int,
    best_key: Optional[Tuple[int, int]],
    global_indices: Set[int],
    scene_db_names: List[str],
    global_db_start: int,
) -> None:
    """打印坐标法 GT 第一帧的全部中间信息。"""
    print(f"\n--- 坐标法 GT 首帧诊断 [{scene_name}] ---")
    print("大图 mapbox geotransform (6 参数):")
    print(f"  gt[0] 左上角 X (投影/经度): {gt[0]}")
    print(f"  gt[1] 像元宽:               {gt[1]}")
    print(f"  gt[2] 旋转(常为 0):         {gt[2]}")
    print(f"  gt[3] 左上角 Y (投影/纬度): {gt[3]}")
    print(f"  gt[4] 旋转(常为 0):         {gt[4]}")
    print(f"  gt[5] 像元高(常为负):       {gt[5]}")
    print(f"大图坐标系: {epsg}")
    print("第一张 query 图片的坐标 (uav_infos 第 1 行):")
    print(f"  经度 longitude = {lon}")
    print(f"  纬度 latitude  = {lat}")
    print("转换为大图投影坐标 (若大图非 WGS84 则先投影):")
    if x_geo is not None and y_geo is not None:
        print(f"  x_geo = {x_geo}")
        print(f"  y_geo = {y_geo}")
    else:
        print("  (未得到)")
    print("逆 geotransform 得到大图像素坐标:")
    print(f"  px (列) = {px}")
    print(f"  py (行) = {py}")
    print("子图尺寸 (从 tif 目录任取一张读取):")
    print(f"  tile_w = {tile_w}, tile_h = {tile_h}")
    if best_key is not None:
        sx, sy = best_key
        print(f"包含 (px,py) 的子图左上角 startx, starty = {sx}, {sy}")
        print("该子图 + 方圆 5x5 邻域对应的全局 DB 索引 (25 张真值):")
        g_sorted = sorted(global_indices)
        print(f"  {g_sorted}")
        print("对应的 db_names 示例 (前 3 个):")
        for idx in g_sorted[:3]:
            local_idx = idx - global_db_start
            if 0 <= local_idx < len(scene_db_names):
                print(f"    db_names[{idx}] = {scene_db_names[local_idx]}")
    else:
        print("未找到包含 (px,py) 的子图，本帧 GT 为空。")
    print("---\n")


def build_gt_9_for_all_trajectories(
    query_trajectory_ranges: List[Tuple[str, int, int]],
    db_scene_ranges: List[Tuple[str, int, int]],
    db_names: List[str],
    datasets_base: Optional[Path] = None,
    verbose: bool = True,
) -> List[Set[int]]:
    """
    为所有 query（按「合并后的 query 列表」顺序）生成 Recall@25 真值集合（每 query 方圆 25 张）。
    query_trajectory_ranges: [(scene_name, start, end), ...]
    db_scene_ranges: [(scene_name, start, end), ...]
    返回 list of set，长度 = 总 query 数；检索命中该集合内任一张即算正确。
    """
    from scene_config import get_scene_paths, DATASETS_BASE

    base = Path(datasets_base) if datasets_base is not None else DATASETS_BASE
    db_scene_by_name = {name: (s, e) for name, s, e in db_scene_ranges}
    out = []
    if verbose:
        print("=== 坐标法 GT@25 各轨迹（未算出 GT 的会打印原因）===")
    for scene_name, q_start, q_end in query_trajectory_ranges:
        n_q = q_end - q_start
        if scene_name not in db_scene_by_name:
            for _ in range(n_q):
                out.append(set())
            if verbose:
                print(f"  [{scene_name}] 共 {n_q} 条 query -> 未算 GT: 该场景不在 DB 中（db_scene_ranges 无此场景）")
            continue
        db_start, db_end = db_scene_by_name[scene_name]
        scene_db_names = db_names[db_start:db_end]
        paths = get_scene_paths(scene_name, base)
        uav_path = paths["uav_infos_path"]
        mapbox_path = paths["mapbox_path"]
        tif_dir = paths["tif_dir"]
        uav_ok = uav_path.exists()
        map_ok = mapbox_path.exists()
        tif_ok = tif_dir.exists()
        if not uav_ok:
            for _ in range(n_q):
                out.append(set())
            if verbose:
                print(f"  [{scene_name}] 共 {n_q} 条 query -> 未算 GT: uav_infos 不存在 ({uav_path})")
            continue
        if not map_ok or not tif_ok:
            for _ in range(n_q):
                out.append(set())
            if verbose:
                print(f"  [{scene_name}] 共 {n_q} 条 query -> 未算 GT: mapbox 存在={map_ok}, tif_dir 存在={tif_ok}")
            continue
        gt_tuple, _ = get_geotransform_and_srs(mapbox_path)
        if gt_tuple is None:
            for _ in range(n_q):
                out.append(set())
            if verbose:
                print(f"  [{scene_name}] 共 {n_q} 条 query -> 未算 GT: 无法读取 mapbox 的 geotransform ({mapbox_path})")
            continue
        tile_wh = get_tile_size_from_tif(Path(tif_dir))
        if tile_wh is None:
            for _ in range(n_q):
                out.append(set())
            if verbose:
                print(f"  [{scene_name}] 共 {n_q} 条 query -> 未算 GT: 无法从 tif 目录读取子图尺寸 ({tif_dir})")
            continue
        # 以上检查都通过，才调用 get_gt_9_for_trajectory（按轨迹 query 数 n_q 对齐；GT 为方圆 25 张）
        list_of_sets = get_gt_9_for_trajectory(
            uav_path,
            scene_db_names,
            mapbox_path,
            tif_dir,
            db_start,
            n_query_expected=n_q,
            scene_name=scene_name,
            debug_first_frame=verbose,
        )
        n_with_gt = sum(1 for s in list_of_sets if len(s) > 0)
        out.extend(list_of_sets)
        if verbose:
            print(f"  [{scene_name}] 共 {n_q} 条 query -> 有 GT 的 {n_with_gt}/{n_q}")
    if verbose:
        total_q = sum(q_end - q_start for _, _, q_end in query_trajectory_ranges)
        total_gt = sum(1 for s in out if len(s) > 0)
        print(f"  -> 合计有 GT 的 query: {total_gt}/{len(out)}\n")
    return out
