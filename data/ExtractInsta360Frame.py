import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None 

VID_DIR = "insta360_video"
OUTPUT_DIR = "ori_images"
CSV_FILE = "Insta360_FrameIndex.csv"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(CSV_FILE):
    print(f"{CSV_FILE}: Not found!!!")
else:
    df_match = pd.read_csv(CSV_FILE)
    for index, row in df_match.iterrows():
        vid_path = os.path.join(VID_DIR, row['Video'])
        frame_idx = int(row['Frame_Index'])
        
        frame_path = os.path.join(OUTPUT_DIR, f"{row['Image']}")

        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(frame_path, frame)
        else:
            print(f"Cannot extract frame {frame_idx} from {vid_path}.")
