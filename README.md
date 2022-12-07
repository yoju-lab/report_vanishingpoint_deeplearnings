### Collactions
#### images google scrapping
- gathering image files in photos folder
- refer : [Github - Google Image Scraper] (https://github.com/ohyicong/Google-Image-Scraper/)
```
pip install -r requirements.txt
python gathering_image_from_google.py   # to datasets/gathering_images/search_keywords directories
```

### Cleanings
#### Remove Outier Images  
- Delete Images with NOT match search_keywords
- handworking Images
#### resize, rotation, crop images 
- resize 200 * 200 pixels 
```
python images_preprocessor.py       # 
```

### EDAs
#### Find Vanishing Points in Images
- refer : [Github - Image Rectification](https://github.com/chsasank/Image-Rectification)
```
python find_vanishing_points.py     # to datasets/find_vanishingpoints and vanishingpoints_infor.csv
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

