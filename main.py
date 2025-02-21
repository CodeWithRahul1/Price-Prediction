"""
hiiiii
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("/home/admin1/Desktop/SmartAppraisal-TRPT/5.jpeg") # pylint: disable=no-member
model = YOLO("/home/admin1/Desktop/SmartAppraisal-TRPT/best.pt")
results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)
for result in results:
    # get array results
    masks = result.masks.data
    boxes = result.boxes.data
    # extract classesimport copy

    clss = boxes[:, 5]
    # get indices of results where class is 0 (people in COCO)
    people_indices = torch.where(clss == 1)
    # use these indices to extract the relevant masks
    people_masks = masks[people_indices]
    # scale for visualizing results
    people_mask = torch.any(people_masks, dim=0).int() * 255
    # save to file
    cv2.imwrite(
        str(
            "/home/admin1/Desktop/SmartAppraisal-TRPT/combined_masks/combined_masks.png"
        ),
        people_mask.cpu().numpy(),
    )


simple_lama = SimpleLama()

# img_path = "/home/admin1/Desktop/SmartAppraisal-TRPT/5.jpeg"
# mask_path = "/home/admin1/Desktop/SmartAppraisal-TRPT/combined_masks/combined_masks.png"

# image = Image.open(img_path)
# mask = Image.open(mask_path).convert('L')

# # Resize both image and mask to the same dimensions
# image = image.resize((512, 512))  # Resize image
# mask = mask.resize((512, 512))  # Resize mask to match

# os.makedirs("output", exist_ok="True")

# result = simple_lama(image, mask)
# result.save("output/inpainted.png")

# plt.figure(figsize=(16, 8))
# plt.subplot(1, 3, 1)
# plt.imshow(image)
# plt.axis("off")
# plt.subplot(1, 3, 2)
# plt.imshow(mask)
# plt.axis("off")
# plt.subplot(1, 3, 3)
# plt.imshow(result)
# plt.axis("off")
# plt.show()



img_path = "/home/admin1/Desktop/SmartAppraisal-TRPT/5.jpeg"
mask_path = "/home/admin1/Desktop/SmartAppraisal-TRPT/combined_masks/combined_masks.png"

# Load images
image = Image.open(img_path)
mask = Image.open(mask_path).convert("L")

# Resize both image and mask to the same dimensions
image = image.resize((512, 512))  # Resize image
mask = mask.resize((512, 512))  # Resize mask to match

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Process the image with the mask (assuming simple_lama is defined elsewhere)
result = simple_lama(image, mask)

# Save the inpainted result
result.save("/home/admin1/Desktop/SmartAppraisal-TRPT/output_image/inpainted.png")

# If you want to still see the images without displaying them, you can save them as well
image.save("/home/admin1/Desktop/SmartAppraisal-TRPT/output_image/original_image.png")
mask.save("/home/admin1/Desktop/SmartAppraisal-TRPT/output_image/mask_image.png")
