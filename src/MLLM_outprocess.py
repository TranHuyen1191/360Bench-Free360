from qwen_vl_utils import process_vision_info
import re
import pdb
import json
import numpy as np
import json
from collections import defaultdict

def get_output_text(model, processor, messages, max_new_tokens=1024):
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    #generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0, do_sample=False)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if 'image_grid_thw' in inputs:
        input_height = inputs['image_grid_thw'][0][1] * 14
        input_width = inputs['image_grid_thw'][0][2] * 14
    else:
        input_height = None
        input_width = None
    return output_text[0], input_width, input_height


def extract_all_coordinates(text, max_per_label=5):
    
    pattern = re.compile(
        r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]'
        r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?'
    )

    matches = pattern.findall(text)

    # Group results per label
    grouped = defaultdict(list)

    for x1, y1, x2, y2, label in matches:
        label = label if label else ""
        grouped[label].append({
            "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
            "label": label
        })

    # Keep only first N per label
    limited = []
    for label, items in grouped.items():
        limited.extend(items[:max_per_label])

    return limited


def extract_all_coordinates_attribute(text, max_per_label=5):
    
    # Regex for bbox with optional label
    pattern = re.compile(
        r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]'
        r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?'
        r'(?:\s*,\s*"attributes"\s*:\s*"([^"]*)")?'
    )

    matches = pattern.findall(text)

    # Group results per label
    grouped = defaultdict(list)

    for x1, y1, x2, y2, label, attributes in matches:
        label = label if label else ""
        attributes = attributes if attributes else ""
        grouped[label].append({
            "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
            "label": label,
            "attributes": attributes
        })

    # Keep only first N per label
    limited = []
    for label, items in grouped.items():
        limited.extend(items[:max_per_label])

    return limited


def process_bboxes(patch_width, patch_height, bounding_boxes_json, patch_width_fromResponse, patch_height_fromResponse, threshold=0):
    bboxes = []
    bbox_labels = []

    # Iterate over the bounding boxes
    for i, bounding_box_item in enumerate(bounding_boxes_json):
        bounding_box = bounding_box_item

        try:
            # --- Extract bbox_2d safely ---
            bbox_2d = bounding_box.get("bbox_2d", [])
            bbox_label = bounding_box.get("label", "")
            if isinstance(bbox_2d, (tuple, list)) and len(bbox_2d) > 0:
                if isinstance(bbox_2d[0], list):
                    bbox_2d = bbox_2d[0]

            if len(bbox_2d) == 4:
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = min(patch_height - 1, int(bbox_2d[1] / patch_height_fromResponse * patch_height))
                abs_x1 = min(patch_width - 1, int(bbox_2d[0] / patch_width_fromResponse * patch_width))
                abs_y2 = min(patch_height - 1, int(bbox_2d[3] / patch_height_fromResponse * patch_height))
                abs_x2 = min(patch_width - 1, int(bbox_2d[2] / patch_width_fromResponse * patch_width))

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1
                if ((abs_y2 - abs_y1) > threshold) and ((abs_x2 - abs_x1) > threshold):
                    bboxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
                    bbox_labels.append(bbox_label)
           
        except Exception as e:
            print("Error processing bounding box:", e, bounding_box)
            continue
    return bboxes, bbox_labels


def process_bboxes_attributes(patch_width, patch_height, bounding_boxes_json, patch_width_fromResponse, patch_height_fromResponse, threshold=0):
    bboxes = []
    bbox_labels = []
    bbox_attributes = []

    # Iterate over the bounding boxes
    for i, bounding_box_item in enumerate(bounding_boxes_json):
        bounding_box = bounding_box_item

        try:
            # --- Extract bbox_2d safely ---
            bbox_2d = bounding_box.get("bbox_2d", [])
            bbox_label = bounding_box.get("label", "")
            bbox_attribute = bounding_box.get("attributes", "")
            if isinstance(bbox_2d, (tuple, list)) and len(bbox_2d) > 0:
                if isinstance(bbox_2d[0], list):
                    bbox_2d = bbox_2d[0]

            if len(bbox_2d) == 4:
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = min(patch_height - 1, int(bbox_2d[1] / patch_height_fromResponse * patch_height))
                abs_x1 = min(patch_width - 1, int(bbox_2d[0] / patch_width_fromResponse * patch_width))
                abs_y2 = min(patch_height - 1, int(bbox_2d[3] / patch_height_fromResponse * patch_height))
                abs_x2 = min(patch_width - 1, int(bbox_2d[2] / patch_width_fromResponse * patch_width))

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1
                if ((abs_y2 - abs_y1) > threshold) and ((abs_x2 - abs_x1) > threshold):
                    bboxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
                    bbox_labels.append(bbox_label)
                    bbox_attributes.append(bbox_attribute)
            #else:
            #    print(abs_x1,abs_x2,abs_y1,abs_y2)
        except Exception as e:
            print("Error processing bounding box:", e, bounding_box)
            continue
    return bboxes, bbox_labels, bbox_attributes

def extract_json_from_markdown(markdown_text, key="object_list"):
    """
    Extract JSON object from a markdown text and return the value of a given key.
    
    Args:
        markdown_text (str): Markdown text containing JSON.
        key (str): Key to extract from the JSON object.
        
    Returns:
        list or None: Value corresponding to the key, or [] if not found.
    """

    # 1. Try to find JSON in ```json ... ``` code block
    pattern_codeblock = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
    matches = pattern_codeblock.findall(markdown_text)

    if matches:
        json_str = matches[-1]
        try:
            data = json.loads(json_str)
            return data.get(key, "")
        except json.JSONDecodeError:
            pass  # fall back to inline search if code block JSON is invalid

    # 2. Fallback: find inline key-value using regex
    # Match either "key": "value" or "key": [ ... ]
    pattern_inline = re.compile(rf'"{re.escape(key)}"\s*:\s*(?:"([^"]*)"|\[([^\]]*)\]|(\w+))')

    match = pattern_inline.search(markdown_text)
    if match:
        if match.group(1):
            # Case 1: string value
            return match.group(1)
        elif match.group(2):
            list_text = match.group(2).strip()
            if "{" in list_text:  # Case 2: list of dict
                if list_text.count("[") > list_text.count("]"):
                    list_text += "]"
                if list_text.count("{") > list_text.count("}"):
                    list_text += "}"

                try:
                    items = json.loads("[" + list_text + "]")
                    return items
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed:", e, list_text)
                    return []
            else:  # Case 2: list value
                items = [item.strip().strip('"') for item in list_text.split(",")]
                return items
        elif match.group(3):
            # Case 3: boolean, number, or null
            val = match.group(3)
            # Optional: convert to correct Python type
            if val.lower() == "true":
                return "true"
            elif val.lower() == "false":
                return "false"
            elif val.lower() == "null":
                return "null"
            else:
                return str(val)
    return ""