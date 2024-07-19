import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# data_path = '/mnt/c/Users/megan/Downloads/Harvesting.v1i.coco/tomato_data'

class ImageAnnotationHandler:
    # Loop through each image and its annotations
    def __init__(self, path) -> None:
        # Read the annotations JSON file
        with open(f'{path}/_annotations.coco.json', 'r') as file:
            data = json.load(file)

        category_id_to_description = {}
        for category in data['categories']:
            category_id = category['id']
            category_id_to_description[category_id] = category['name']

        annotations = data['annotations']
        image_id_to_annotations = {}
        for annotation in annotations:
            image_id = annotation['image_id']
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
        self.category_id_to_description = category_id_to_description
        self.path = path
      
    def plot_image(self, image_ids, rows = 3, columns = 3):
        
        # Create a plot
        fig, axes = plt.subplots(rows, columns, figsize=(10,10))
        
        # Display the image
        for i, ax in enumerate(axes.flat):
            image_id = image_ids[i]
            image_path = self.path + '/' + self.image_id_to_image_path[image_id]
            image = mpimg.imread(image_path)
            ax.imshow(image)
            ax.axis('off')

            # Add annotations
            annotations = self.image_id_to_annotations[image_id]
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

if __name__ == '__main__':
    train_handler = ImageAnnotationHandler('tomato_data/train')
    valid_handler = ImageAnnotationHandler('tomato_data/valid')

    train_handler.plot_image([0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(f'train is plotted')

    valid_handler.plot_image(list(range(9)))
    print(f'valid is plotted')