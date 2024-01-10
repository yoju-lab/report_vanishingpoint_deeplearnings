```
pip install -r requirements.txt
```
### Collactions
#### scrapping images from google  (needs to check it)
- gathering image files in photos folder
- refer : [Github - Google Image Scraper] (https://github.com/ohyicong/Google-Image-Scraper/)
```
~$ python gathering_image_from_google.py   # to datasets/gathering_images/search_keywords directories
```
#### from bing
- refer : [colab](https://colab.research.google.com/drive/1iu9Jwp45n8p15aF29qmehykKP6HLtJgx)
```
~$ python Collactions/gathering_images_from _bing.py   # to datasets/collections/fromBings/
```

### Preprocessings
#### resize, rotation, crop images 
- resize 200 * 200 pixels and covert gray scale
```
python images_preprocessor.py       # from datasets/collections to datasets/preprocessings
```
#### delete image_similarity
```
~$ python image_similarity.py   # to datasets/any_informations/delete_images.csv, target_images_withoutsimilar.csv
```

### EDAs
#### Find Vanishing Points in Images
- refer : [Github - Image Rectification](https://github.com/chsasank/Image-Rectification)
```
python find_vanishing_points.py     # to datasets/find_vanishingpoints and vanishingpoints_infor.csv
```
#### Remove Outlier Images  
- Delete Images with NOT match search_keywords
- handworking Images
```
python filtering_use_grayimage.py     # to datasets/any_informations/filtering_grayimage_by_vanishing_point.csv
```

### ModelBuildings
#### Datasets split
- handworking Images : training, validation directories - 8:2
#### Model Training
- Fine Tuning Model : ResNet50, InceptionResNetV2
- save model with trained
- export history files
```
python finetuning_InceptionResNetV2.py      # export h5 model and history files
python finetuning_ResNet50.py               # export h5 model and history files
```
#### ModelDeployments
- load model with trained
- predictions with images
```
python model_deplpyments.py
```

