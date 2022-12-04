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
### ?
```
python find_vanishingpoints_deeplearning.py

```

####

