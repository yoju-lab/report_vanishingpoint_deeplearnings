import json
import os


if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from commons import utils

    configs_path = 'commons/configs.json'
    configs = json.load(open(configs_path))

    collections_images_path = os.path.join(
        configs['datasets_dir'], configs['gathering_images_dir'])
    # images_sizes, images_df = utils.get_images_information(
    #     collections_images_path)
    # print(images_df)
    images_list = utils.load_image_paths_from_folder(collections_images_path)
    pass

    # get height, width from images
    # min_height, min_width = utils.get_min_size(images_sizes)
    min_height, min_width = tuple(configs['resize_image_height_width'])
    pass
    # resize images
    utils.resize_images(images_list, min_height, min_width)
    pass
