# 360Bench-Free360
## Project Structure

```text
root/
├── data/
│   ├── ori_images                        # Directory for raw/extracted images
│   ├── insta360_video                    # Directory for raw video downloads
│   ├── ERP_images                        # Directory for preprocessed 7K ERP images
│   ├── CMP4x3_images                        # Directory for CMP images (converted from images in ERP_images)
│   ├── ExtractInsta360Frame.py           # Extracts frames from Insta360 videos and save to ori_images
│   ├── ImagePreprocess.py                # Downsample to 7K
│   ├── 360Bench.tsv                      # Main benchmark annotations
│   ├── Flickr_Noirlab_Links.csv          # Metadata for Flickr/Noirlab images
│   └── Insta360_VideoLinks.csv           # Metadata for Insta360 images 
│   └── Insta360_ImageLinks.csv           # Metadata for Insta360 videos
│   └── Insta360_FrameIndex.csv           # Metadata for Insta360 video frame extraction
└── 360lib/                               # [360Lib Software for ERP to CMP Conversion](https://vcgit.hhi.fraunhofer.de/jvet/360lib/-/blob/360Lib-13.4/360Lib_README.txt?ref_type=tags)
└── src/                                  # Source codes of Free360 Method  
└── Free360.py                            # Our proposed Free360 framework  
```

---
## 360Bench Dataset Preparation 
### Image/Video Download
- Images: Download images from Flickr_Noirlab_Links.csv and Insta360_ImageLinks.csv. Store all files in the data/ori_images/ directory.
- Videos: Download videos from Insta360_VideoLinks.csv and store them in the data/insta360_video/ directory.

### Insta360 Frame Extraction
- Use ExtractInsta360Frame.py to process the downloaded videos. The script references the mapping in Insta360_FrameIndex.csv to extract specific frames, which are then saved directly into the data/ori_images/ directory.

### Image Preprocessing
- Use ImagePreprocess.py to downsample all images in the data/ori_images/ directory to a 7K resolution (7296 x 3648 pixels) and then store them in the data/ERP_images/ directory.
- Download and Install 360Lib following instructions (i.e., Step 1: VTM-19.0-360Lib-13.4 software preparation) at [360Lib](https://vcgit.hhi.fraunhofer.de/jvet/360lib/-/blob/360Lib-13.4/360Lib_README.txt?ref_type=tags) 
- Use ConvertERP2CMP.py to convert ERP images to CMP images using 360Lib. 

### 360Bench Benchmark
360Bench.tsv is the main annotation file of the benchmark. 
Each row corresponds to a multiple-choice question associated with a 360° image.
Each entry includes:
- mage_file: image filename
- question: question about the image
- category / l2-category: task type (e.g., SR-Os)
- bbox: bounding boxes of relevant objects in ERP images
- multi-choice options: multiple-choice options
- answer: correct answer

### Free360 framework 
To execute the Free360 framework, run the main Python script:

```bash
python Free360.py
```

## Authors
- Huyen Tran - *Tohoku University, Japan*
- Van-Quang Nguyen - *RIKEN, Japan*
- Farros Alferro - *Tohoku University, Japan*
- Kang-Jun Liu - *Tohoku University, Japan*
- Takayuki Okatani - *Tohoku University, Japan*
  
_If you have any questions or comments, please feel free to contact me via tranhuyen1191@gmail.com._ 


## Acknowledgments
If you use this dataset in your research, please cite the reference below:
```
   @inproceedings{tran2026360degimageperceptionmllms,
        title={360{\deg} Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method}, 
        author={Huyen T. T. Tran and Van-Quang Nguyen and Farros Alferro and Kang-Jun Liu and Takayuki Okatani},
        year={2026},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        eprint={2603.16179},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2603.16179}, 
    }
```
## License
* Images and Videos: All visual media are subject to the original licenses provided by their respective sources (e.g., Flickr/Noirlab and Insta360). Users must adhere to the specific terms set by the original providers.
* Annotations: The questions, categories, and bounding box data provided in 360Bench.tsv are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
 
