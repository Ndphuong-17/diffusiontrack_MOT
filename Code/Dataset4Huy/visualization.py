from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def original_shape(track, img_w, img_h):
    if len(track.shape) == 1:
        track = np.expand_dims(track, axis=0)

    track[:, 0::2] = track[:, 0::2] * img_w
    track[:, 1::2] = track[:, 1::2] * img_h
    track[:, 0] = track[:, 0] - track[:, 2] / 2
    track[:, 1] = track[:, 1] - track[:, 3] / 2

    return np.round(track, decimals=1)


def visualize_bbox(image_path, bbox):
    """
    Visualize a bounding box on an image.

    Parameters:
    - image_path (str): Path to the image file.
    - bbox (list or tuple): Bounding box coordinates in the format [x, y, width, height],
      where (x, y) is the top-left corner of the bounding box.
    """
    # Open the image
    image = Image.open(image_path)

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Handle the case where bbox might be a 2D array
    if hasattr(bbox, "shape") and len(bbox.shape) == 2:
        bbox = bbox[0]

    # Calculate the bottom-right corner of the bounding box
    x, y, width, height = bbox
    x2, y2 = x + width, y + height

    # Draw the bounding box
    draw.rectangle([x, y, x2, y2], outline="red", width=2)

    # # Show the image
    image.show()


    # Display the image with matplotlib
    plt.imshow(image)
    plt.axis("off")  # Hide axis for a cleaner look
    plt.show()