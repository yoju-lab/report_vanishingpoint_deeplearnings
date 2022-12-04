# -*- coding: utf-8 -*-
from GoogleImageScraper import GoogleImageScraper
import concurrent.futures
import os
import sys
sys.path.append('./')


def worker_thread(search_key):
    image_scraper = GoogleImageScraper(
        webdriver_path, image_path, search_key, number_of_images, headless, min_resolution, max_resolution)
    image_urls = image_scraper.find_image_urls()
    image_scraper.save_images(image_urls, keep_filenames)

    # Release resources
    del image_scraper


if __name__ == "__main__":
    import json
    configs_path = 'commons/configs.json'
    configs = json.load(open(configs_path))
    # Define file path
    webdriver_dir = configs['webdriver_dir']

    from commons.patch import webdriver_executable
    webdriver_path = os.path.normpath(os.path.join(
        os.getcwd(), webdriver_dir, webdriver_executable()))

    datasets_dir = configs['datasets_dir']
    gathering_images_dir = configs['gathering_images_dir']
    image_path = os.path.normpath(os.path.join(
        os.getcwd(), datasets_dir, gathering_images_dir))
    import os
    os.makedirs(image_path, exist_ok=True)

    search_list = configs["search_keywords"]
    search_keys = list(set(search_list))

    # Parameters
    # Desired number of images
    number_of_images = configs["number_of_images"]
    headless = False                     # True = No Chrome GUI
    # headless = True                     # True = No Chrome GUI
    min_resolution = (0, 0)             # Minimum desired image resolution
    max_resolution = (9999, 9999)       # Maximum desired image resolution
    max_missed = 1000                   # Max number of failed images before exit
    number_of_workers = 1               # Number of "workers" used
    keep_filenames = False              # Keep original URL image filenames

    # Run each search_key in a separate thread
    # Automatically waits for all threads to finish
    # Removes duplicate strings from search_keys
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        executor.map(worker_thread, search_keys)
