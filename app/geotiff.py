import rasterio
from rasterio.plot import show
fp = r'data/sat_data/nust/nust.tif'
img = rasterio.open(fp)
print(img.count)
print(img.height, img.width)
print(img.crs)
show(img, transform=img.transform)


def reproject_raster(in_path, out_path, crs):

    """
    """
    # reproject raster to project crs
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = rasterio.calculate_default_transform(src_crs, crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height})

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=rasterio.Resampling.nearest)
    return(out_path)
