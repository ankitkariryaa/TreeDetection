import geopandas as gps
from fire import Fire
from tqdm import tqdm
import string
import os

def map_grid_x_index(total_cuts, cuts_x, p):
    coord = list(p.coords[0])
    x = -1
    for idx, val in enumerate(cuts_x[::-1]):
        if coord[0] > val:
            x = total_cuts - idx
            break
    return(x)

def map_grid_y_index(total_cuts, cuts_y, p):
    coord = list(p.coords[0])
    y = -1
    for idx, val in enumerate(cuts_y[::-1]):
        if coord[1] > val:
            y = total_cuts - idx
            break
    return(y)

def split_dataset(input_file, output_path, division_count = 4):
    """
    Adds image_idx field to each annotation in input geojson and writes the output geojson file.
    Script may fail if output file already exists.
    :param input_file: Path to geojson file.
    :param output_path: Path for output geojson files.
    :param division_count: Number of divisons to perform (32)
    """
    gdf = gps.read_file(input_file)
    total_bounds = gdf['geometry'].total_bounds
    gx = total_bounds[2] - total_bounds[0]
    gy = total_bounds[3] - total_bounds[1]
    total_cuts = division_count // 2
    cuts_x = [((total_bounds[0] + (i / total_cuts) * gx) // 1000) * 1000 for i in range(total_cuts)]
    cuts_y = [((total_bounds[1] + (i / total_cuts) * gy) // 1000) * 1000 for i in range(total_cuts)]
    gdf['reference_grid_x'] = gdf.apply(lambda row: map_grid_x_index(total_cuts, cuts_x, row['geometry']), axis=1)
    gdf['reference_grid_y'] = gdf.apply(lambda row: map_grid_y_index(total_cuts, cuts_y, row['geometry']), axis=1)
    lx = string.ascii_uppercase[:total_cuts]
    for idx, x in enumerate(range(total_cuts)):
        for y in range(total_cuts):
            pdf = gdf[(gdf['reference_grid_x'] == x + 1) & (gdf['reference_grid_y'] == y + 1)]
            out_file = os.path.join(output_path,'trees_{}_{}.geojson'.format(lx[x], y + 1))
            # print(out_file)
            if not pdf.empty:
                pdf.to_file(out_file, driver='GeoJSON')

# Usage: python3 -m detection.dataset.dataset_split /Users/kari/Vision/Dataset/Hamburg/subset/trees_out.geojson \
#  /Users/kari/Vision/Dataset/Hamburg/subset/

if __name__ == '__main__':
    Fire(split_dataset)
