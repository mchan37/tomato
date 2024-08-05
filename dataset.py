import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2
import numpy as np
import os

# data_path = '/mnt/c/Users/megan/Downloads/Harvesting.v1i.coco/tomato_data'

class ImageAnnotationHandler:
    # Loop through each image and its annotations
    def __init__(self, path, ignore_categories = None) -> None:
        if ignore_categories is None:
            ignore_categories = {0} # default ignore tomato

        # Read the annotations JSON file
        with open(f'{path}/_annotations.coco.json', 'r') as file:
            data = json.load(file)

        category_id_to_description = {}
        for category in data['categories']:
            category_id = category['id']
            if category_id in ignore_categories:
                continue
            category_id_to_description[category_id] = category['name']
        
        category_id_to_label_id = {}
        for idx, k in enumerate(category_id_to_description.keys()):
            category_id_to_label_id[k] = idx

        annotations = data['annotations']
        image_id_to_annotations = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            if category_id in ignore_categories:
                continue

            image_annotations = image_id_to_annotations.get(image_id)
            if image_annotations is None:
                image_annotations = []
                image_id_to_annotations[image_id] = image_annotations
            image_annotations.append(annotation)

        image_id_to_image_path = {}
        for image in data['images']:
            image_id = image['id']
            image_id_to_image_path[image_id] = image['file_name']

        self.image_id_to_image_path = image_id_to_image_path
        self.image_id_to_annotations = image_id_to_annotations
        image_id_with_no_annotations = set(self.image_id_to_image_path.keys()) - set(self.image_id_to_annotations.keys())
        self.image_id_to_image_path = {k: v for k, v in self.image_id_to_image_path.items() if k not in image_id_with_no_annotations}
        self.category_id_to_description = category_id_to_description
        self.category_id_to_label_id = category_id_to_label_id
        self.path = path
        self.data = data

    def num_categories(self):
        return len(self.category_id_to_description)
      
    def plot_image(self, image_ids, rows = 3, columns = 3):
        if len(image_ids) < rows * columns:
            rows = 1
            columns = len(image_ids)
    
        # Create a plot
        fig, axes = plt.subplots(rows, columns, figsize=(10,10))
        
        # Display the image
        if len(image_ids) == 1:
            all_axes = [axes]
        else:
            all_axes = axes.flat

        for i, ax in enumerate(all_axes):
            image_id = image_ids[i]
            image_path = self.path + '/' + self.image_id_to_image_path[image_id]
            image = mpimg.imread(image_path)
            ax.imshow(image)
            ax.axis('off')

            # Add annotations
            annotations = self.image_id_to_annotations.get(image_id)
            if annotations is None:
                continue
            for annotation in annotations:
                bbox = annotation['bbox']
                label = self.category_id_to_description[annotation['category_id']]
                # Create a rectangle patch
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
                # Add label text
                ax.text(bbox[0], bbox[1] - 10, label, color='red', fontsize=10, weight='bold')
        
        # Show the plot
        plt.tight_layout(pad = 0.1)
        plt.show()

    def sift_features(self, image_ids = None, rows = 3, columns = 3, show = False):
        # default None gives you sift_features for all images
        if image_ids is None:
            image_ids = list(self.image_id_to_image_path.keys())

        # Create a plot
        if show:
            fig, axes = plt.subplots(rows, columns, figsize=(10,10))
        
        sift_features = []
        labels = []
        
        # Display the image
        for i, image_id in enumerate(image_ids):
            image_path = self.path + '/' + self.image_id_to_image_path[image_id]
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            annotations = self.image_id_to_annotations.get(image_id)
            if annotations:
                labels.append(self.category_id_to_label_id[annotations[0]['category_id']]) # pick first label
            else:
                labels.append(-1)
            
            # Create a SIFT detector
            sift = cv2.SIFT_create()

            # Detect keypoints and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            if descriptors is not None:
                sift_features.append(descriptors)
            else:
                sift_features.append(np.zeros((1, 128)))   

            if show:
                # image index to row, column
                ax = axes[i // columns, i % columns]
                # Draw keypoints on the image
                img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # Display the image with keypoints in the grid
                ax.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                # ax.set_title(image_path)

        # Display the image with keypoints
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        if show:
            plt.show()

        return sift_features, labels


if __name__ == '__main__':
    train_handler = ImageAnnotationHandler('tomato_data/train')
    valid_handler = ImageAnnotationHandler('tomato_data/valid')

    train_handler.sift_features([0, 1, 2, 3, 4, 5, 6, 7, 8], show = True)
    print(f'train sift_features is plotted')

    features = valid_handler.sift_features(list(range(9)))
    print(f'valid sift_features is of length {len(features)}')

    train_handler.plot_image([0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(f'train is plotted')

    valid_handler.plot_image([568])
    print(f'valid is plotted')