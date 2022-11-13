from use_utils import utils
            
images_sizes, images_df = utils.get_images_information()            
# print(images_df)
pass

# get height, width from images
min_height, min_width = utils.get_min_size(images_sizes)
pass
# resize images 
utils.resize_images(images_df, min_height, min_width)
pass