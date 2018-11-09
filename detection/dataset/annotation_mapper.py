import geopandas as gps
from fire import Fire
from tqdm import tqdm


def add_image_idxs(input_file, output_file, zone=32):
    """
    Adds image_idx field to each annotation in input geojson and writes the output geojson file.
    Script may fail if output file already exists.
    :param input_file: Path to geojson file.
    :param output_file: Path to output geojson file.
    :param zone: UTF zone (32)
    """
    gdf = gps.read_file(input_file)
    print("Read annotations from input geojson:{}".format(input_file))
    all_rows = []
    for idx, row in tqdm(gdf.iterrows(), desc='Processing:', total=gdf.shape[0]):
        p = row['geometry']
        coord = list(p.coords[0])
        img_idx = int(coord[0] // 1000), int(coord[1] // 1000)
        img_id = 'dop20rgb_{}{}_{}_1_hh_2016.jpg'.format(zone, img_idx[0], img_idx[1])
        row['image_idx'] = img_id
        all_rows.append(row)
    gdf = gps.geodataframe.GeoDataFrame(all_rows)
    gdf.to_file(output_file, driver="GeoJSON")
    print("Written output to file:{}".format(output_file))


# Usage: python3 -m detection.dataset.annotation_mapper /home/sanjeev/Downloads/subset/trees.geojson \
#  /home/sanjeev/Downloads/subset/trees_out.geojson

if __name__ == '__main__':
    Fire(add_image_idxs)
