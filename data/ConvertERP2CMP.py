import subprocess
import os 
from PIL import Image
import numpy as np
import pdb
import pandas as pd

def convertErpRgb2Cmp(image_path,CMP_dir,image_idx,converter,CMP_type):
    image = Image.open(image_path).convert('RGB') 
    wdt, hgt = image.size
    yuv_path = os.path.join(CMP_dir, image_idx + f'_{wdt}x{hgt}.yuv')
    rgb_frame = np.asarray(image)
    W_face = int(wdt/4)
    H_face = int(hgt/2)
    CMP_W= W_face*4
    CMP_H= H_face*3

    def rgb_to_yuv(rgb_frame,yuv_path):
        matrix = np.array( [[ 65.481/255,     128.553/255,     24.966/255],
                            [-37.797/255,    -74.203/255,      112.0/255],
                            [ 112.0/255,     -93.786/255,     -18.214/255]])

        matrix = matrix.T
        yuv_frame = np.matmul(rgb_frame, matrix) + np.array([[16, 128, 128]], dtype="uint8")
        Y = np.clip(yuv_frame[:, :, 0], 16, 235)
        U = np.clip(yuv_frame[:, :, 1], 16, 240)
        V = np.clip(yuv_frame[:, :, 2], 16, 240)   

        # Downsample U and V to 1/2 width and 1/2 height (simple average pooling)
        def downsample_2x(channel):
            return ((channel[0::2, 0::2] + channel[1::2, 0::2] +
                     channel[0::2, 1::2] + channel[1::2, 1::2]) / 4)
        Y = Y.astype(np.uint8)
        U = (downsample_2x(U)).astype(np.uint8)
        V = (downsample_2x(V)).astype(np.uint8)
        with open(yuv_path, "wb") as f:
            f.write(Y.tobytes())  # Y plane
            f.write(U.tobytes())  # U plane
            f.write(V.tobytes())  # V plane

    def yuv420p_to_rgb_pil(yuv_path,width,height, jpg_path):
        with open(yuv_path, 'rb') as f:
            yuv = np.frombuffer(f.read(), dtype=np.uint8)

        frame_size = width * height
        uv_size = frame_size // 4

        # Extract Y, U, V
        Y = yuv[0:frame_size].reshape((height, width))
        U = yuv[frame_size:frame_size + uv_size].reshape((height // 2, width // 2))
        V = yuv[frame_size + uv_size:].reshape((height // 2, width // 2))


        # Upsample U and V to match Y size
        U_up = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1)
        V_up = np.repeat(np.repeat(V, 2, axis=0), 2, axis=1)

        # Convert YUV to RGB manually
        Y = Y.astype(np.float32)
        U = U_up.astype(np.float32) - 128
        V = V_up.astype(np.float32) - 128

        R = Y + 1.402 * V
        G = Y - 0.344136 * U - 0.714136 * V
        B = Y + 1.772 * U

        rgb = np.stack((R, G, B), axis=-1)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        rgb_img = Image.fromarray(rgb, 'RGB')
        rgb_img.save(jpg_path)

    
    if CMP_type == "3x2":
        CodingFPStructure = "2 3 4 0 0 0 5 0 3 180 1 270 2 0"
    else:
        CodingFPStructure = "3 4 2 90 6 0 7 0 8 0  1 0 4 0 0 0 5 0 3 270 9 0 10 0 11 0"
    output_yuv_CMP = os.path.join(CMP_dir, image_idx + f'.yuv')
    output_jpg_CMP = os.path.join(CMP_dir, image_idx + f'.jpg')
    if os.path.exists(output_jpg_CMP):
        print(f"{output_jpg_CMP} existed")
    else:
        yuv_frame = rgb_to_yuv(rgb_frame,yuv_path)
        cmd = [
            converter,
            "-i", yuv_path,
            "-wdt", f"{wdt}",
            "-hgt", f"{hgt}",
            "--InputBitDepth=8",
            "--OutputBitDepth=8",
            "-fs", "0",
            "-icf", "420",
            "--SourceFPStructure=1 1 0 0",
            "--InputGeometryType=0",
            "--CodingGeometryType=1",
            f"--CodingFPStructure={CodingFPStructure}",
            f"--CodingFaceWidth={W_face}", 
            f"--CodingFaceHeight={H_face}", 
            "-f", "1",
            "-o", output_yuv_CMP
        ]

        output_yuv_CMP = os.path.join(CMP_dir, image_idx + f'_{CMP_W}x{CMP_H}_0Hz_8b_420.yuv')

        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        
        yuv420p_to_rgb_pil(output_yuv_CMP,CMP_W,CMP_H,output_jpg_CMP)
        os.remove(output_yuv_CMP)

        os.remove(yuv_path)


dataset_dir = "./" 
dataset = "360VQA_2_onlyA"
CMP_type = "4x3" # "3x2" or "4x3"
CMP_dir = f'./CMP{CMP_type}_images'
converter = "./../UbuntuTohokuDesk/360lib/VVCSoftware_VTM/bin/360ConvertAppStatic"
os.makedirs(CMP_dir, exist_ok=True)
image_dir = './images2'


data_path = os.path.join(dataset_dir, f"{dataset}.tsv")
data = pd.read_csv(data_path, sep='\t') 
for index, item in data.iterrows():
    image_idx = os.path.splitext(item["image_file"])[0]  
    image_path = os.path.join(image_dir, item["image_file"])
    output_jpg = os.path.join(CMP_dir, image_idx + '.jpg')
    if os.path.exists(output_jpg):
        print(f"{output_jpg} existed")
    else:
        convertErpRgb2Cmp(image_path,CMP_dir,image_idx,converter,CMP_type)


