# 360Bench-Free360
## Project Structure

```text
root/
├── data/
│   ├── ori_images                        # Directory for raw/extracted images
│   ├── insta360_video                    # Directory for raw video downloads
│   ├── ERP_images                        # Directory for preprocessed 7K ERP images
│   ├── ExtractInsta360Frame.py           # Extracts frames from Insta360 videos and save to ori_images
│   ├── ImagePreprocess.py                # Downsample to 7K
│   ├── 360Bench.tsv                      # Main benchmark annotations
│   ├── Flickr_Noirlab_Links.csv          # Metadata for Flickr/Noirlab images
│   └── Insta360_VideoLinks.csv           # Metadata for Insta360 images 
│   └── Insta360_ImageLinks.csv           # Metadata for Insta360 videos
│   └── Insta360_FrameIndex.csv           # Metadata for Insta360 video frame extraction
└── Source codes of Free360 Method       # (To be published upon acceptance)
```

---
## 360Bench Dataset Preparation 
### Image/Video Download
- Images: Download images from Flickr_Noirlab_Links.csv and Insta360_ImageLinks.csv. Store all files in the ori_images/ directory.
- Videos: Download videos from Insta360_VideoLinks.csv and store them in the insta360_video/ directory.

### Insta360 Frame Extraction
- Use the ExtractInsta360Frame.py script to process the downloaded videos. The script references the mapping in Insta360_FrameIndex.csv to extract specific frames, which are then saved directly into the ori_images/ directory.

### Image Preprocessing
- Downsample all images within the ori_images/ directory to a 7K resolution ($7296 \times 3648$ pixels) and then store them in the ERP_images/ directory.

### Benchmark Annotation: 
360Bench.tsv is the main annotation file of the benchmark. Each row corresponds to a multiple-choice question associated with a 360° image.

Each entry includes:
#### image_file: image filename
#### question: question about the image
#### category / l2-category: task type (e.g., SR-Os)
#### bbox: bounding boxes of relevant objects in ERP images
#### multi-choice options: multiple-choice options
#### answer: correct answer

## Authors

* **Huyen Tran** - *Tohoku University, Japan* - tranhuyen1191@gmail.com
* **Van-Quang Nguyen** - *RIKEN, Japan*
* **Farros Alferro** - *Tohoku University, Japan*
* **Kang-Jun Liu** - *Tohoku University, Japan*
* **Takayuki Okatani** - *Tohoku University, Japan* 

* If you have any questions or comments, please feel free to contact me via tranhuyen1191@gmail.com. 


## Acknowledgments
If you use this source code in your research, please cite the references below:

   @misc{tran2026360degimageperceptionmllms,
        title={360{\deg} Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method}, 
        author={Huyen T. T. Tran and Van-Quang Nguyen and Farros Alferro and Kang-Jun Liu and Takayuki Okatani},
        year={2026},
        eprint={2603.16179},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2603.16179}, 
    }

## License

### Images and Videos: All visual media are subject to the original licenses provided by their respective sources (e.g., Flickr/Noirlab and Insta360). Users must adhere to the specific terms set by the original providers.

### Annotations: The questions, categories, and bounding box data provided in 360Bench.tsv are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
 
