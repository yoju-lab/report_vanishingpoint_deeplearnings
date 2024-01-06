import os
from PIL import Image
import imagehash
import pandas as pd

# Function to calculate the similarity hash between two images
def calculate_image_similarity(image_path1, image_path2):
    hash1 = imagehash.average_hash(Image.open(image_path1))
    hash2 = imagehash.average_hash(Image.open(image_path2))
    return hash1 - hash2

# Function to find similar images and delete duplicates
def find_and_delete_similar_images(directory, similarity_threshold=10):
    # Create a DataFrame to store image paths and similarity degrees
    similar_images_df = pd.DataFrame(columns=['Image1', 'Image2', 'Similarity'])

    # Get a list of image files in the specified directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Iterate through all pairs of images and calculate similarity
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            image1_path = os.path.join(directory, image_files[i])
            image2_path = os.path.join(directory, image_files[j])

            similarity = calculate_image_similarity(image1_path, image2_path)

            # Append data to the DataFrame
            similar_images_df = similar_images_df.append({'Image1': image1_path, 'Image2': image2_path, 'Similarity': similarity}, ignore_index=True)

    csv_path = os.path.join('datasets/any_informations/target_images_withoutsimilar.csv')
    # Filter images with similarity above the threshold
    target_images_withoutsimilar_df = similar_images_df[similar_images_df['Similarity'] > similarity_threshold]
    target_images_withoutsimilar_df.to_csv(csv_path, index=False)

    # Filter images with similarity above the threshold
    delete_images_df = similar_images_df[similar_images_df['Similarity'] <= similarity_threshold]

    # Create image_similarity.csv to store similarity degree
    csv_path = os.path.join('datasets/any_informations/delete_images.csv')
    delete_images_df.to_csv(csv_path, index=False)

    # Delete all but one of those with a similarity of 0.9 or higher
    delete_file_list = []
    for index, row in delete_images_df.iterrows():
        try:
            delete_file_list.append(row['Image2'])
            # os.remove(row['Image2'])
        except Exception as e:
            print(f"Error deleting {row['Image2']}: {e}")
            
    print('len(image_files):{}, len(delete_file_list):{}'.format(len(image_files),
                                                                 len(delete_file_list)))
    
if __name__ == '__main__':
    # Set the path to the images directory
    image_directory = 'datasets/preprocessings'

    # Set the similarity threshold
    similarity_threshold = 0.9

    # Find and delete similar images
    find_and_delete_similar_images(image_directory, similarity_threshold)