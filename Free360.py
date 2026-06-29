import os
import re
import json
import time
import torch
import pandas as pd
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image

from src.prompts import TEMPLATE_PROMPTS
from src.view_relations import VIEW_RELATIONS_LIST,view_idx,filter_relations
from src.data_process import read_data
from src.MLLM_outprocess import get_output_text, extract_all_coordinates,process_bboxes,extract_json_from_markdown
from src.bbox_filter import is_considered
from src.projection_format import coor2faceidx_v1,generate_masks_prompt_v3
from src.evaluation import evaluate, get_dimension_rating


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

cudnn.benchmark = False
cudnn.deterministic = True

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
MLLM_name = "Qwen/Qwen2.5-VL-7B-Instruct"
face_W = 1824
face_H = 1824
erp_W, erp_H = 7296, 3648

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--modelname", type=str, default="Free360_v1")
parser.add_argument("--annotation_file", type=str, default="360Bench.tsv")
parser.add_argument("--pro_format", type=str, default="CMP4x3")
parser.add_argument("--max_obj_per_face", type=int, default=5)
parser.add_argument("--chunk_idx", type=int, default=0)
parser.add_argument("--total_chunks", type=int, default=1)
parser.add_argument("--max_pixels", type=int, default=7296 * 5472)
parser.add_argument("--ROI_Gen", type=int, default=3)
parser.add_argument("--CropFlag", type=str2bool, default=True)
parser.add_argument("--EVRel", type=str2bool, default=True)

args = parser.parse_args()
view_rels = VIEW_RELATIONS_LIST["short"]

if args.pro_format == "ERP":
    img_W, img_H = 7296, 3648
else:
    img_W, img_H = 7296, 5472



proj_dir = os.path.expanduser(".")
print(f"Project directory: {proj_dir}")

data_dir = os.path.join(proj_dir, "data")
print(f"Data directory: {data_dir}")

image_dir = os.path.join(data_dir, f'{args.pro_format}_images')  # ERP_images, CMP4x3_images
ERPimage_dir = os.path.join(data_dir, 'ERP_images')  # ERP_images, CMP4x3_images
cubefaces_dir = os.path.join(data_dir, "cubefaces")

cf = "" if args.CropFlag else "_woCrop"
evrel = "" if args.EVRel else "_woEVRel"
output_dir = os.path.join(
    "results", args.modelname +
    f"_{args.pro_format}_{args.max_obj_per_face}_{args.ROI_Gen}{cf}{evrel}")

print(output_dir)

ROI_dir = os.path.join(output_dir, "ROI")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(ROI_dir, exist_ok=True)

result_file = os.path.join(output_dir, f"{args.modelname}_rating_chunk{args.chunk_idx}.json")
pred_file = os.path.join(output_dir, f"{args.modelname}_pred_chunk{args.chunk_idx}.csv")
log_file = os.path.join(output_dir, f"{args.modelname}_log_chunk{args.chunk_idx}.txt")

if os.path.exists(pred_file):
    print(f"###########################################\n File {pred_file} exists. --> Continue from this file!")
    saved_data = pd.read_csv(pred_file)
    data = saved_data
else:
    # Read data
    data = read_data(data_dir, image_dir, args.annotation_file, args.chunk_idx, args.total_chunks)

data['ERPimage_path'] = data['image_file'].apply(lambda p: os.path.join(ERPimage_dir, p))

# Load template prompt
system_prompt = "You are a helpful assistant"
template_prompt = TEMPLATE_PROMPTS

# Load model and processor
processor = AutoProcessor.from_pretrained(MLLM_name, use_fast=True, max_pixels=args.max_pixels)

MLLM = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MLLM_name,
    dtype="bfloat16",
    device_map="cuda",
    attn_implementation='flash_attention_2',
)

device = next(MLLM.parameters()).device
# Main loop
for index, item in data.iterrows():
    if "prediction" in data.columns and pd.notnull(item.get("prediction", None)):
        continue  # Skip already processed entries
    options = eval(item["multi-choice options"])
    rawquestion = item["question"]
    question = rawquestion + "\t" + "\t".join(options)
    image_path = item["image_path"]
    ERPimage_path = item["ERPimage_path"]
    image_file = item["image_file"]

    question_idx = item["index"]
    question_type = item["category"] + "/" + item["l2-category"]
    response = ""
    sample_start_time = time.time()

    detectview = range(6)
    full_bboxes = []
    full_objs = []
    full_att_des = []
    full_rel_des = []
    act_views = []

    img = Image.open(image_path)
    bboxes = []
    answer, width_fromResponse, height_fromResponse = get_output_text(MLLM, processor, [{
        "role": "system",
        "content": system_prompt
    }, {
        "role":
            "user",
        "content": [{
            "type": "text",
            "text": template_prompt["objdetection"].format(rawquestion)
        }, {
            "image": image_path
        }]
    }])
    
    answer_bboxes = extract_all_coordinates(answer, args.max_obj_per_face)
    bboxes, bbox_labels = process_bboxes(img_W, img_H, answer_bboxes, width_fromResponse, height_fromResponse)
    
    #Group bboxes by label
    grouped = defaultdict(list)
    for bbox, label in zip(bboxes, bbox_labels):
        grouped[label].append(bbox)
    grouped = dict(grouped)

    selected_boxes = []
    selected_objs = []
    for bbox_label, bboxes in grouped.items():
        selected_boxes_per_label = []
        # filter bboxes and get coressponding descriptions
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = [
                max(x1 - float(0.3) * (x2 - x1),0), 
                max(y1 - float(0.3) * (y2 - y1),0), 
                min(x2 + float(0.3) * (x2 - x1), img_W-1),
                min(y2 + float(0.3) * (y2 - y1),img_H-1)
            ]
            global_bbox = [x1, y1, x2, y2]
            # if bbox is not overlapped with previously detected objs --> adding it CHECK
            if is_considered(global_bbox, selected_boxes_per_label, threshold=0.5):
                selected_boxes.append(global_bbox)
                selected_boxes_per_label.append(global_bbox)
                faceidx = coor2faceidx_v1(global_bbox,img_W,img_H)
                full_bboxes.append(global_bbox)
                obj_label = bbox_label + f" {len(selected_boxes_per_label)}"
                assert obj_label not in full_objs
                assert obj_label not in selected_objs 
                full_objs.append(obj_label)
                selected_objs.append(obj_label)

                if args.EVRel:
                    full_rel_des.append(obj_label + "-->in-->" + view_idx[faceidx])
                if view_idx[faceidx] not in act_views:
                    act_views.append(view_idx[faceidx])

                obj_label_ = re.sub(r'[^A-Za-z0-9._#-]+', '_', obj_label)

                if args.CropFlag: 

                    ROI_img = img.crop((x1, y1, x2, y2))
                    ROI_path = os.path.join(ROI_dir, f"{question_idx}_ROI_{obj_label_}_face{faceidx}_{len(selected_boxes)}.jpg")
                    ROI_img.save(ROI_path)

                    caption_attobj_json, _, _ = get_output_text(MLLM, processor, [{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role":
                            "user",
                        "content": [{
                            "type": "text",
                            "text": template_prompt["Caption_Attributes"].format(bbox_label, rawquestion)
                        }, {
                            "image": ROI_path
                        }]
                    }])
                else:
                    
                    caption_attobj_json, _, _ = get_output_text(MLLM, processor, [{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role":
                            "user",
                        "content": [{
                            "type": "text",
                            "text": template_prompt["Caption_Attributes"].format(bbox_label, rawquestion)
                        }, {
                            "image": image_path
                        }]
                    }])

                caption_attobj = extract_json_from_markdown(caption_attobj_json, "Description")
                caption_attobj = caption_attobj.replace("The image shows ", "")

                caption_text = extract_json_from_markdown(caption_attobj_json, "Text")
                
                if isinstance(caption_text, list):
                    caption_text = ", ".join(caption_text)
                if len(caption_attobj) > 0:
                    connect_word = ". TEXT: " if caption_attobj[-1] != "." else " TEXT: "
                else:
                    connect_word = " TEXT: "
                full_att_des.append(caption_attobj + connect_word + caption_text if len(caption_text) > 0 else caption_attobj)

    pair_num = 0
    need_rotation = False
    if len(selected_boxes) > 0:  # have new objects detected --> generate caption of spatial relationship
        for i, (new_bbox, new_obj) in enumerate(zip(selected_boxes, selected_objs)):
            lef_new_bboxes = selected_boxes[i + 1:]
            lef_obj_bboxes = selected_objs[i + 1:]

            for pre_bbox, pre_obj in zip(lef_new_bboxes, lef_obj_bboxes):
                ROI_path = os.path.join(ROI_dir, f"{question_idx}_ROI_Comb{pair_num}.jpg")
                pair_box = [new_bbox, pre_bbox]
                pair_cap = [new_obj, pre_obj]
                masks_str,need_rotation_ = generate_masks_prompt_v3(ERPimage_path,image_path,
                                                        pair_cap, pair_box, 
                                                        ROI_path,img_W,img_H,erp_W,erp_H,face_W,face_H,device,args.ROI_Gen)
                #print(masks_str)
                if need_rotation_:
                    need_rotation = True
                
                
                if masks_str is not None:
                    try:
                        caption_sparel, _, _ = get_output_text(MLLM, processor, [{
                            "role": "system",
                            "content": system_prompt
                        }, {
                            "role":
                                "user",
                            "content": [{
                                "type":
                                    "text",
                                "text":
                                    template_prompt["Caption_spatialrelation"].format(
                                        question,
                                        pair_cap,
                                        masks_str,
                                    )
                            }, {
                                "image": ROI_path
                            }]
                        }])
                        full_rel_des.append(extract_json_from_markdown(caption_sparel, "output"))
                        
                        pair_num = pair_num + 1
                    except Exception as e:
                        pass
                    
                # Delete the ROI image after use
                """
                try:
                    os.remove(ROI_path)
                except OSError:
                    pass
                """

    scene_graph, json_response = "", ""

    act_att_views, act_rel_views = filter_relations(act_views, view_rels)
    full_objs.extend(act_views)
    full_att_des.extend(act_att_views)
    if args.EVRel:
        full_rel_des.extend(act_rel_views)

    list_nodes = ", ".join(full_objs)
    list_att = "\n\t".join([f"{obj}-->{att}" for (obj, att) in zip(full_objs, full_att_des)])
    list_rel = ", ".join(full_rel_des)
    scene_graph = f"List of Nodes: {list_nodes}\nAttribute relations:\n\t{list_att}\nSpatial relations: {list_rel}"

    
    json_response, _, _ = get_output_text(MLLM, processor, [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": [{
            "type": "text",
            "text": template_prompt["Final_answer"].format(question, scene_graph)
        }, {
            "image": image_path
        }]
    }])

    response = extract_json_from_markdown(json_response, "answer")

    if response == "CANNOT ANSWER":

        response, _, _ = get_output_text(MLLM, processor, [{
            "role":
                "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {
                    "type":
                        "text",
                    "text":
                        f"{question}. Select the best answer to the given multiple-choice question based on the image. Respond with only the letter (A, B, C, or D) of the correct option.\nThe best answer is:"
                },
            ],
        }])

    if response == "":
        response = "CANNOT ANSWER!"

    sample_end_time = time.time()
    sample_duration = sample_end_time - sample_start_time
    data.loc[index, "infTime"] = sample_duration

    data.loc[index, "prediction"] = response
    data.loc[index, "scene_graph"] = scene_graph
    data.loc[index, "json_response"] = str(json_response)
    data.loc[index, "full_bboxes"] = str(full_bboxes)
    data.loc[index, "full_objs"] = str(full_objs)
    data.loc[index, "need_rotation"] = str(need_rotation)

    data.to_csv(pred_file, index=False)

# Check if infTime column exists before accessing it
if "infTime" in data.columns:
    valid_inf_times = data["infTime"].dropna()
    total_time = valid_inf_times.sum()
    average_time_per_iteration = valid_inf_times.mean()
else:
    total_time = 0
    average_time_per_iteration = 0

with open(log_file, "a") as logfile:
    print(f"############ Total time taken for {len(data):.0f} iterations: {total_time:.4f} seconds", file=logfile)
    print(f"############ Average time per iteration: {average_time_per_iteration:.6f} seconds", file=logfile)

# Compute Accuracy only if DataFrame is not empty
if len(data) > 0 and "prediction" in data.columns:
    data,num_error,num_rejected = evaluate(data)
    rating = get_dimension_rating(data)

    output = {"rating": rating}

    # Save as JSON
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(rating)

    data.to_csv(pred_file, index=False)
else:
    print(f"Warning: No data to evaluate in chunk {args.chunk_idx}. Skipping evaluation.")
    # Still save empty/incomplete results
    if len(data) > 0:
        data.to_csv(pred_file, index=False)
