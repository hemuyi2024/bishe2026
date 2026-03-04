from osgeo import gdal, osr
import sys

# 检查是否提供了文件路径参数
if len(sys.argv) != 2:
    print("用法: python script_name.py <影像文件路径>")
    sys.exit(1)

# 获取命令行参数中的影像文件路径
raster_file = sys.argv[1]

# 打开遥感影像文件
dataset = gdal.Open(raster_file)
if dataset is None:
    print("无法打开影像文件。")
else:
    # 获取地理变换参数和投影信息
    geotransform = dataset.GetGeoTransform()  # 主要是需要这个
    print("地理变换参数:", geotransform)
    #save the geotransform
    with open("village_geotransform.txt", "w") as f:
        for value in geotransform:
            f.write(f"{value}\n")
    proj = dataset.GetProjection()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj)
    print("投影信息:", spatial_ref)
    # 打印投影坐标信息 (Proj4 格式)
    if spatial_ref:
        proj4_str = spatial_ref.ExportToProj4()
        print(f"Proj4 投影坐标信息: {proj4_str}")
    else:
        print("未找到投影坐标信息。")

    # 示例：将像素坐标转换为地理坐标（经纬度）
    pixel_x, pixel_y = 5, 10  # 示例像素坐标
    if geotransform:
        origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
        geo_x = origin_x + pixel_x * pixel_width
        geo_y = origin_y + pixel_y * pixel_height

        # 将地理坐标转换为经纬度
        coord_transform = osr.CoordinateTransformation(spatial_ref, spatial_ref.CloneGeogCS())
        lon, lat, _ = coord_transform.TransformPoint(geo_x, geo_y)

        print(f"像素坐标 ({pixel_x}, {pixel_y}) 转换为经纬度坐标: 经度 {lon}, 纬度 {lat}")
        print(f"像素坐标 ({pixel_x}, {pixel_y}) 转换为地理坐标: x={geo_x}, y={geo_y}")
