from PIL import Image
import os
import pandas as pd
import pdb

Image.MAX_IMAGE_PIXELS = None  
def resize_images(image_path, output_path, width, height):
    with Image.open(image_path) as img:
        or_width, or_height = img.size
        if or_width/or_height == 2 and or_width>=7296:
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
            resized_img.save(output_path)
            print(f"Resized: {image_path} -> {output_path}")


resize_width = 7296
resize_height = 3648

INPUT_DIR = './ori_images'
OUTPUT_DIR = './ERP_images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        resize_images(image_path, output_path, resize_width, resize_height)

