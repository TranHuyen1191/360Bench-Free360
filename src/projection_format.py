import numpy as np
from PIL import Image
import torch
import cv2
import numpy as np
import math

S_PI = math.pi
S_EPS = 1e-8  # small epsilon to avoid division by zero
S_PI_2 = math.pi / 2

# Define distinct colors (BGR) for masks
color_palette = [
    ((0,0,255), "red"),
    ((255,0,0), "blue"),
    ((0,255,0), "green"),
    ((0,255,255), "yellow"),
    ((255,0,255), "magenta"),
    ((255,255,0), "cyan"),
    ((128,0,128), "purple"),
    ((0,128,128), "teal"),
    ((128,128,0), "olive"),
]
    

def coor2faceidx(x,y,img_W,img_H):
    if y<img_H/3: faceidx = 2  # top
    elif y>=2*img_H/3: faceidx = 3  # bottom
    else:
        if x<img_W/4: faceidx = 1  # behind
        elif x<img_W/2: faceidx = 4  # left
        elif x<3*img_W/4: faceidx = 0  # front
        else: faceidx = 5  # right
    return faceidx


def coor2faceidx_v1(global_bbox,img_W,img_H):
    x1, y1, x2, y2 = global_bbox 
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    
    faceidx_center = coor2faceidx(bbox_center_x,bbox_center_y,img_W,img_H)
 
    return faceidx_center

def generate_masks_prompt(image_path, entities, boxes, mask_path):
    """
    Args:
        image_path (str): Path to the input image.
        entities (list of str): List of entity names.
        boxes (list of tuples): List of bounding boxes [(x1, y1, x2, y2), ...] corresponding to entities.
        mask_path (str): Path to save the masked image.
        
    Returns:
        masks_str (str): String describing entity-mask colors for prompt.
    """
    # Load image
    img = cv2.imread(image_path)


    # Crop image to union
    masked_img = img.copy()
    # Define distinct colors (BGR) for masks
    color_palette = [
        ((0,0,255), "red"),
        ((255,0,0), "blue"),
        ((0,255,0), "green"),
        ((0,255,255), "yellow"),
        ((255,0,255), "magenta"),
        ((255,255,0), "cyan"),
        ((128,0,128), "purple"),
        ((0,128,128), "teal"),
        ((128,128,0), "olive"),
    ]
    
    masks_str_list = []

    # Sort boxes by area (smallest first)
    areas = [(i, (box[2]-box[0])*(box[3]-box[1])) for i, box in enumerate(boxes)]
    areas.sort(key=lambda x: x[1])  # ascending order
    
    for idx, _ in areas:
        entity = entities[idx]
        box = boxes[idx]
        
        color, color_name = color_palette[idx % len(color_palette)]
        
        # Adjust box coordinates relative to cropped image
        x1, y1, x2, y2 = map(int, box)
        
        # Define box coordinates (relative)
        pt1 = (x1, y1)
        pt2 = (x2, y2)

        # Draw rectangle (bounding box) with color boundary
        cv2.rectangle(masked_img, pt1, pt2, color, thickness=3)

        # Prepare mask string
        masks_str_list.append(f"{entity}: {color_name} line")
    
    # Save masked image
    cv2.imwrite(mask_path, masked_img)
    
    return masks_str_list


def CMP4x3coor2facecoor(x, y, img_W, img_H, face_W, face_H):
    """
    Convert CMP 4x3 coordinates (x, y) to face coordinates.
    Works with scalar or tensor x, y (PyTorch tensors).
    """
    # Initialize result tensors
    faceidx = torch.empty_like(x, dtype=torch.long)
    face_x = torch.empty_like(x, dtype=torch.float)
    face_y = torch.empty_like(y, dtype=torch.float)

    # --- Conditions ---
    top_mask = y < img_H / 3
    bottom_mask = y >= 2 * img_H / 3
    behind_mask = (~top_mask) & (~bottom_mask) & (x < img_W / 4)
    left_mask = (~top_mask) & (~bottom_mask) & (x >= img_W / 4) & (x < img_W / 2)
    front_mask = (~top_mask) & (~bottom_mask) & (x >= img_W / 2) & (x < 3 * img_W / 4)
    right_mask = (~top_mask) & (~bottom_mask) & (x >= 3 * img_W / 4)

    # --- Top face ---
    faceidx[top_mask] = 2
    face_x[top_mask] = face_H - y[top_mask]
    face_y[top_mask] = x[top_mask]

    # --- Bottom face ---
    faceidx[bottom_mask] = 3
    y_b = y[bottom_mask] - 2 * face_H
    face_x[bottom_mask] = y_b
    face_y[bottom_mask] = face_W - x[bottom_mask]

    # --- Behind face ---
    faceidx[behind_mask] = 1
    face_x[behind_mask] = x[behind_mask]
    face_y[behind_mask] = y[behind_mask] - face_H

    # --- Left face ---
    faceidx[left_mask] = 4
    face_x[left_mask] = x[left_mask] - face_W
    face_y[left_mask] = y[left_mask] - face_H

    # --- Front face ---
    faceidx[front_mask] = 0
    face_x[front_mask] = x[front_mask] - 2 * face_W
    face_y[front_mask] = y[front_mask] - face_H

    # --- Right face ---
    faceidx[right_mask] = 5
    face_x[right_mask] = x[right_mask] - 3 * face_W
    face_y[right_mask] = y[right_mask] - face_H

    return faceidx, torch.clamp(face_x, min=0), torch.clamp(face_y, min=0)

def map2DCMPTo3D_batch(x, y, faceIdx, face_width, face_height):
    """
    Convert 2D cubemap face coordinates to 3D unit vectors on a sphere (batched version).

    Args:
        x: tensor of x coordinates, shape (N,)
        y: tensor of y coordinates, shape (N,)
        faceIdx: tensor of face indices, shape (N,)
        face_width: width of one cubemap face
        face_height: height of one cubemap face

    Returns:
        x3d, y3d, z3d: tensors of 3D coordinates, shape (N,)
    """
    u = x + 0.5
    v = y + 0.5

    # Normalize to [-1, 1]
    pu = 2.0 * u / face_width - 1.0
    pv = 2.0 * v / face_height - 1.0

    # Initialize output tensors
    x3d = torch.zeros_like(pu)
    y3d = torch.zeros_like(pv)
    z3d = torch.zeros_like(pu)

    # Face 0
    mask = faceIdx == 0
    x3d[mask] = 1.0
    y3d[mask] = -pv[mask]
    z3d[mask] = -pu[mask]

    # Face 1
    mask = faceIdx == 1
    x3d[mask] = -1.0
    y3d[mask] = -pv[mask]
    z3d[mask] = pu[mask]

    # Face 2
    mask = faceIdx == 2
    x3d[mask] = pu[mask]
    y3d[mask] = 1.0
    z3d[mask] = pv[mask]

    # Face 3
    mask = faceIdx == 3
    x3d[mask] = pu[mask]
    y3d[mask] = -1.0
    z3d[mask] = -pv[mask]

    # Face 4
    mask = faceIdx == 4
    x3d[mask] = pu[mask]
    y3d[mask] = -pv[mask]
    z3d[mask] = 1.0

    # Face 5
    mask = faceIdx == 5
    x3d[mask] = -pu[mask]
    y3d[mask] = -pv[mask]
    z3d[mask] = -1.0

    # Optional: check invalid indices
    if torch.any((faceIdx < 0) | (faceIdx > 5)):
        raise ValueError("Invalid faceIdx in map2DCMPTo3D_batch")

    return x3d, y3d, z3d

def map3DToLongLat_batch(X, Y, Z, shift_range_to_2pi=True):
    """
    Convert 3D coordinates (X, Y, Z) to longitude and latitude.
    Supports PyTorch tensors or scalars.
    Longitude range: [0, 2π) if shift_range_to_2pi=True, otherwise [-π, π).
    Latitude range: [-π/2, π/2].
    """
    # Compute longitude: atan2(-Z, X)
    Long = torch.atan2(-Z, X)

    if shift_range_to_2pi:
        Long = torch.where(Long < 0, Long + 2 * math.pi, Long)

    # Compute normalization safely
    norm = torch.sqrt(X**2 + Y**2 + Z**2 + 1e-12)  # avoid divide-by-zero
    Lat = torch.asin(torch.clamp(Y / norm, -1.0, 1.0))

    return Long, Lat

def mapLongLatTo3D(longitude, latitude):
    """
    Convert longitude and latitude to 3D coordinates on the unit sphere.
    Args:
        longitude: angle in radians
        latitude: angle in radians
    Returns:
        tuple (x, y, z) representing the 3D coordinates on the unit sphere.
    """
    x = math.cos(latitude) * math.cos(longitude)
    y = math.sin(latitude)
    z = -math.cos(latitude) * math.sin(longitude)
    return x, y, z

def map2DERPTo3D(x,y, erp_width, erp_height):
    """
    Adapted from 360Lib
    Source: https://vcgit.hhi.fraunhofer.de/jvet/360lib.git
    """
    u = x + 0.5
    v = y + 0.5

    if (u < 0 or u >= erp_width) and (0 <= v < erp_height):
        u = erp_width + u if u < 0 else u - erp_width
    elif v < 0:
        v = -v
        u = u + (erp_width / 2)
        u = u - erp_width if u >= erp_width else u
    elif v >= erp_height:
        v = (2 * erp_height) - v
        u = u + (erp_width / 2)
        u = u - erp_width if u >= erp_width else u

    longitude = u * 2 * S_PI / erp_width - S_PI
    latitude = S_PI_2 - v * S_PI / erp_height

    x,y,z = mapLongLatTo3D(longitude, latitude)
    return x,y,z

def map3DToCMP_single(X,Y,Z, face_width, face_height):
    aX = abs(X)
    aY = abs(Y)
    aZ = abs(Z)

    pu, pv = 0.0, 0.0

    if aX >= aY and aX >= aZ:
        if X > 0:
            faceIdx = 0
            pu = -Z / aX
            pv = -Y / aX
        else:
            faceIdx = 1
            pu = Z / aX
            pv = -Y / aX
    elif aY >= aX and aY >= aZ:
        if Y > 0:
            faceIdx = 2
            pu = X / aY
            pv = Z / aY
        else:
            faceIdx = 3
            pu = X / aY
            pv = -Z / aY
    else:
        if Z > 0:
            faceIdx = 4
            pu = X / aZ
            pv = -Y / aZ
        else:
            faceIdx = 5
            pu = -X / aZ
            pv = -Y / aZ

    x = (pu + 1.0) * (face_width / 2.0) - 0.5
    y = (pv + 1.0) * (face_height / 2.0) - 0.5

    return x,y,faceIdx

def facecoor2CMP4x3coor(faceidx, face_x, face_y, img_W, img_H, face_W, face_H):
    """
    Convert face coordinates back to CMP 4x3 coordinates.
    No tensor version, works with scalar values.

    faceidx:
        0 = front
        1 = behind
        2 = top
        3 = bottom
        4 = left
        5 = right
    """

    if faceidx == 0:  # Front
        x = face_x + 2 * face_W
        y = face_y + face_H

    elif faceidx == 1:  # Behind
        x = face_x
        y = face_y + face_H

    elif faceidx == 2:  # Top
        y = face_H - face_x
        x = face_y

    elif faceidx == 3:  # Bottom
        y = face_x + 2 * face_H
        x = face_W - face_y

    elif faceidx == 4:  # Left
        x = face_x + face_W
        y = face_y + face_H

    elif faceidx == 5:  # Right
        x = face_x + 3 * face_W
        y = face_y + face_H

    else:
        raise ValueError(f"Invalid faceidx: {faceidx}")

    # optional clamp to image boundary
    x = max(0, min(x, img_W - 1))
    y = max(0, min(y, img_H - 1))

    return x, y


def get_covering_rect(Long, Lat):
    """
    Compute minimal rectangular bounds (min/max longitude & latitude)
    that cover all points. Handles wrap-around 
    """

    if not torch.is_tensor(Long):
        Long = torch.tensor(Long, dtype=torch.float)
    if not torch.is_tensor(Lat):
        Lat = torch.tensor(Lat, dtype=torch.float)

    # --- Latitude range (straightforward) ---
    min_lat, max_lat = Lat.min(), Lat.max()

    # --- Longitude range (with wrap-around handling) ---
    # Sort and compute gaps
    sorted_long, _ = torch.sort(Long)
    diffs = torch.diff(sorted_long)
    # Include wrap-around gap
    wrap_gap = (sorted_long[0] + 2 * math.pi) - sorted_long[-1]
    diffs = torch.cat([diffs, wrap_gap.unsqueeze(0)])

    # Find largest gap → minimal rectangle avoids this gap
    max_gap_idx = torch.argmax(diffs)
    start_idx = (max_gap_idx + 1) % len(sorted_long)

    # Shift so that region starts after the largest gap
    shifted = torch.cat([sorted_long[start_idx:], sorted_long[:start_idx] + 2 * math.pi])

    min_long = shifted[0]
    max_long = shifted[-1]

    # Normalize back to [0, 2π)
    min_long = min_long % (2 * math.pi)
    max_long = max_long % (2 * math.pi)

    return min_long, max_long, min_lat, max_lat

def compute_center_and_size(min_long, max_long, min_lat, max_lat, ERPimg_W, ERPimg_H):
    # Handle longitude wrap-around
    if min_long > max_long:
        # Wrap-around case
        RoICenterLong = ((min_long + max_long + 2*S_PI) / 2) % (2*S_PI)
        ROI_W = int(ERPimg_W/(2*S_PI) * ((max_long - min_long + 2*S_PI) % (2*S_PI)))
    else:
        RoICenterLong = (min_long + max_long) / 2
        ROI_W = int(ERPimg_W/(2*S_PI) * (max_long - min_long))
    
    RoICenterLat = (min_lat + max_lat) / 2
    ROI_H = int(ERPimg_H/S_PI * (max_lat - min_lat))
    
    return RoICenterLong, RoICenterLat, ROI_W, ROI_H

def RotMCalculation_tensor(RoICenterLong, RoICenterLat):
    """
    Compute rotation matrix around Y-axis (longitude) and Z-axis (latitude).
    Inputs are in radians. Works with PyTorch tensors.
    """
    cosL = torch.cos(RoICenterLong)
    sinL = torch.sin(RoICenterLong)
    cosB = torch.cos(RoICenterLat)
    sinB = torch.sin(RoICenterLat)

    # Rotation around Y-axis by longitude
    RotM_y = torch.tensor([
        [cosL, 0.0, sinL],
        [0.0, 1.0, 0.0],
        [-sinL, 0.0, cosL]
    ], dtype=torch.float32, device=RoICenterLong.device if torch.is_tensor(RoICenterLong) else None)

    # Rotation around Z-axis by latitude
    RotM_z = torch.tensor([
        [cosB, -sinB, 0.0],
        [sinB,  cosB, 0.0],
        [0.0,   0.0,  1.0]
    ], dtype=torch.float32, device=RoICenterLat.device if torch.is_tensor(RoICenterLat) else None)

    # Combined rotation: first rotate around Z, then around Y
    RotM = RotM_y @ RotM_z
    return RotM


def map3DTo2DERP_batch(x, y, z, erp_width, erp_height):
    """
    Adapted from 360Lib
    Source: https://vcgit.hhi.fraunhofer.de/jvet/360lib.git
    Convert 3D point on unit sphere to 2D equirectangular projection.

    Args:
        erp_width: width of the output image
        erp_height: height of the output image

    Returns:
        dict with keys 'x', 'y', 'z', 'faceIdx'
    """

    # longitude angle maps to horizontal coordinate
    x_out = (S_PI - torch.atan2(z, x)) * erp_width / (2 * S_PI) - 0.5
    length = torch.sqrt(x**2 + y**2 + z**2)
    
    # Avoid division by zero for very small lengths
    mask = length < S_EPS
    safe_div = torch.where(mask, torch.tensor(0., device=length.device), y / length)
    safe_div = torch.clamp(safe_div, -1.0, 1.0)

    # Latitude angle maps to vertical coordinate
    y_out = torch.acos(safe_div) * erp_height / S_PI
    y_out = torch.where(mask, 0.5 * erp_height, y_out)

    y_out = y_out - 0.5

    return torch.clamp(x_out,0,erp_width-1),  torch.clamp(y_out,0,erp_height-1)

def mapLongLatTo3D_batch(longitude, latitude):
    """
    Convert longitude and latitude to 3D coordinates on the unit sphere.
    Args:
        longitude: angle in radians
        latitude: angle in radians
    Returns:
        tuple (x, y, z) representing the 3D coordinates on the unit sphere.
    """
    # Map to 3D unit sphere
    x3d = torch.cos(latitude) * torch.cos(longitude)
    y3d = torch.sin(latitude)
    z3d = -torch.cos(latitude) * torch.sin(longitude)
    return x3d, y3d, z3d


def map2DOfsetERPTo3D_batch(x, y, erp_width, erp_height, offset_x):
    """
    Adapted from 360Lib
    Source: https://vcgit.hhi.fraunhofer.de/jvet/360lib.git

    Map 2D ERP coordinates to 3D points on the unit sphere,
    with the projection center shifted from I(0, 0, 0) to O(-offset_x,0,0).

    Args:
        x (Tensor): 1D tensor of horizontal pixel coordinates.
        y (Tensor): 1D tensor of vertical pixel coordinates.
        erp_width (int): Width of the ERP image.
        erp_height (int): Height of the ERP image.
        offset_x (float): X offset of the new projection center.

    Returns:
        Tensor: (N, 3) tensor of 3D points on the unit sphere.
    """
    u = x + 0.5
    v = y + 0.5

    # Wrap u and v coordinates as in 360Lib
    cond_u = (u < 0) | (u >= erp_width)
    cond_v_lower = v < 0
    cond_v_upper = v >= erp_height

    u = torch.where(cond_u & (v >= 0) & (v < erp_height), (u + erp_width) % erp_width, u)
    u = torch.where(cond_v_lower | cond_v_upper, (u + erp_width / 2) % erp_width, u)
    v = torch.where(cond_v_lower, -v, v)
    v = torch.where(cond_v_upper, 2 * erp_height - v, v)


    longitude = u * 2 * S_PI / erp_width - S_PI
    latitude = S_PI_2 - v * S_PI / erp_height

    # Compute X, Y, Z of P in I frame
    # P is the point on the unit sphere corresponding to (longitude, latitude)
    X_I,Y_I,Z_I = mapLongLatTo3D_batch(longitude, latitude)    
    P_I = torch.stack([X_I, Y_I, Z_I], dim=1)  

    # Transform P to O frame
    P_O = P_I + torch.tensor([offset_x, 0.0, 0.0], device=P_I.device)

    # Compute I's coordinates in O frame
    I_O = torch.tensor([offset_x, 0.0, 0.0], device=P_I.device).view(1, 3).expand_as(P_O)

    # Find intersection between line I_O -> P_O and unit sphere centered at O
    direction = P_O - I_O  # (N, 3)
    direction = direction / torch.norm(direction, dim=1, keepdim=True)  # normalize
    
    # Solve |I_O + t * direction|^2 = 1 for t
    a = (direction ** 2).sum(dim=1)
    b = 2 * (I_O * direction).sum(dim=1)
    c = (I_O ** 2).sum(dim=1) - 1

    discriminant = b ** 2 - 4 * a * c
    t = (-b + torch.sqrt(discriminant)) / (2 * a)  # use the farther intersection

    # Intersection point on the unit sphere (in O frame)
    Q = I_O + t.unsqueeze(1) * direction  # (N, 3)
    # Q is the intersection point on the unit sphere in the O frame

    return Q[:,0],Q[:,1],Q[:,2]  # Shape: (N, 3)

def createCMPRoIfromCMP4x3BBoxes(entities, ROI_path, corners_x, corners_y,
                                   CMPimg_W, CMPimg_H, ERPimg_W, ERPimg_H,
                                   face_W, face_H, ERPimagepath, device,offset=0):
    
    # --- Load ERP image once, as tensor on device ---
    image = Image.open(ERPimagepath).convert('RGB')
    old_rgb_erp = torch.from_numpy(np.array(image)).to(device)  # (H,W,3)
    
    # --- CMP -> face coordinates ---
    faceidx, face_x, face_y = CMP4x3coor2facecoor(
        corners_x,
        corners_y,
        CMPimg_W, CMPimg_H, face_W, face_H
    )
    
    # --- Map to 3D, then to ERP ---
    X_3D, Y_3D, Z_3D = map2DCMPTo3D_batch(face_x, face_y, faceidx, face_W, face_H)
    Long, Lat = map3DToLongLat_batch(X_3D, Y_3D, Z_3D, shift_range_to_2pi=True)
    
    # --- Get ROI rectangle in ERP ---
    min_long, max_long, min_lat, max_lat = get_covering_rect(Long, Lat)
    RoICenterLong, RoICenterLat, ROI_W, ROI_H = compute_center_and_size(min_long, max_long, min_lat, max_lat, ERPimg_W, ERPimg_H)

    #RoICenterLong, RoICenterLat = torch.tensor([0]).to(device),torch.tensor([0]).to(device)
    # --- Rotation matrices ---
    RotMnew2old = RotMCalculation_tensor(RoICenterLong, RoICenterLat)
    RotMold2new = torch.linalg.inv(RotMnew2old)
    
    # --- Rotate 3D points ---
    old_3D = torch.stack([X_3D, Y_3D, Z_3D], dim=1).float()
    new_3D = (RotMold2new @ old_3D.T).T
    new_2D_x, new_2D_y = map3DTo2DERP_batch(new_3D[:,0], new_3D[:,1], new_3D[:,2], ERPimg_W, ERPimg_H)
    
    #new_2D_x = torch.clamp(new_2D_x - (ERPimg_W-ROI_W)/2, 0, ROI_W)
    #new_2D_y = torch.clamp(new_2D_y - (ERPimg_H-ROI_H)/2, 0, ROI_H)

    # --- Create ERP grid (vectorized) ---
    xs = torch.arange(0, ERPimg_W, device=device)
    ys = torch.arange(0, ERPimg_H, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    grid_x = grid_x.flatten().float()
    grid_y = grid_y.flatten().float()
    
    # --- Map ERP grid to 3D ---
    new_3D_x, new_3D_y, new_3D_z = map2DOfsetERPTo3D_batch(grid_x, grid_y, ERPimg_W, ERPimg_H, offset)
    new_3D_grid = torch.stack([new_3D_x, new_3D_y, new_3D_z], dim=1).float()
    
    # --- Rotate back and sample old ERP pixels ---
    old_3D_grid = (RotMnew2old @ new_3D_grid.T).T
    old_2D_x, old_2D_y = map3DTo2DERP_batch(old_3D_grid[:,0], old_3D_grid[:,1], old_3D_grid[:,2], ERPimg_W, ERPimg_H)
    old_2D_x = old_2D_x.clamp(0, ERPimg_W-1).long()
    old_2D_y = old_2D_y.clamp(0, ERPimg_H-1).long()
    sampled_pixels = old_rgb_erp[old_2D_y, old_2D_x]
    new_rgb_erp = sampled_pixels.view(ERPimg_H, ERPimg_W, 3).cpu().numpy().astype(np.uint8)
    
    
    masks_str_list = []
    n_entities = len(entities)
    # --- Draw all bounding boxes efficiently ---
    img_np = cv2.cvtColor(new_rgb_erp, cv2.COLOR_RGB2BGR)

    for idx in range(n_entities):
        entity = entities[idx]
        color, color_name = color_palette[idx % len(color_palette)]
        start, end = idx*4, (idx+1)*4
        x1_rel, y1_rel = int(new_2D_x[start:end].min()), int(new_2D_y[start:end].min())
        x2_rel, y2_rel = int(new_2D_x[start:end].max()), int(new_2D_y[start:end].max())
        cv2.rectangle(img_np, (x1_rel, y1_rel), (x2_rel, y2_rel), color, 30)
        masks_str_list.append(f"{entity}: {color_name} line")
    # --- Save final ERP ROI image ---
    cv2.imwrite(ROI_path, img_np)
    
    return masks_str_list


def get_corner_coordinates_torch_v2(boxes, img_W, img_H, device="cuda"):
    boxes = torch.tensor(boxes, dtype=torch.float32, device=device)

    # corners: [x1,y1,x1,y2,x2,y1,x2,y2] for all boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners_x = torch.stack([x1, x1, x2, x2], dim=1).flatten()
    corners_y = torch.stack([y1, y2, y1, y2], dim=1).flatten()

    # delta masks
    delta_x = corners_x - img_W / 4
    delta_y_top = img_H / 3 - corners_y
    delta_y_bottom = corners_y - 2 * img_H / 3

    # --- top boundary adjustment ---
    top_mask = (corners_y < img_H / 3) & (corners_x >= img_W / 4)
    mask = top_mask 
    corners_y[mask] = img_H / 3
    

    # --- bottom boundary adjustment ---
    bottom_mask = (corners_y >= 2 * img_H / 3) & (corners_x >= img_W / 4)
    mask = bottom_mask
    corners_y[mask] = 2 * img_H / 3 - 1
    

    # --- clamp to image bounds ---
    corners_x = torch.clamp(corners_x, 0, img_W - 1)
    corners_y = torch.clamp(corners_y, 0, img_H - 1)

    # --- compute faceid ---
    faceid = torch.empty_like(corners_x, dtype=torch.long)
    top = corners_y < img_H / 3
    bottom = corners_y >= 2 * img_H / 3
    behind = (~top) & (~bottom) & (corners_x < img_W / 4)
    left = (~top) & (~bottom) & (corners_x >= img_W / 4) & (corners_x < img_W / 2)
    front = (~top) & (~bottom) & (corners_x >= img_W / 2) & (corners_x < 3 * img_W / 4)
    right = (~top) & (~bottom) & (corners_x >= 3 * img_W / 4)

    faceid[top] = 2
    faceid[bottom] = 3
    faceid[behind] = 1
    faceid[left] = 4
    faceid[front] = 0
    faceid[right] = 5

    return corners_x, corners_y, faceid

def createRoIfromCMP4x3BBoxes(entities, ROI_path, corners_x, corners_y,
                                   CMPimg_W, CMPimg_H, ERPimg_W, ERPimg_H,
                                   face_W, face_H, ERPimagepath, device,offset=0):
    
    # --- Load ERP image once, as tensor on device ---
    image = Image.open(ERPimagepath).convert('RGB')
    old_rgb_erp = torch.from_numpy(np.array(image)).to(device)  # (H,W,3)
    
    # --- CMP -> face coordinates ---
    faceidx, face_x, face_y = CMP4x3coor2facecoor(
        corners_x,
        corners_y,
        CMPimg_W, CMPimg_H, face_W, face_H
    )
    
    # --- Map to 3D, then to ERP ---
    X_3D, Y_3D, Z_3D = map2DCMPTo3D_batch(face_x, face_y, faceidx, face_W, face_H)
    Long, Lat = map3DToLongLat_batch(X_3D, Y_3D, Z_3D, shift_range_to_2pi=True)
    
    # --- Get ROI rectangle in ERP ---
    min_long, max_long, min_lat, max_lat = get_covering_rect(Long, Lat)
    RoICenterLong, RoICenterLat, ROI_W, ROI_H = compute_center_and_size(min_long, max_long, min_lat, max_lat, ERPimg_W, ERPimg_H)

    #RoICenterLong, RoICenterLat = torch.tensor([0]).to(device),torch.tensor([0]).to(device)
    # --- Rotation matrices ---
    RotMnew2old = RotMCalculation_tensor(RoICenterLong, RoICenterLat)
    RotMold2new = torch.linalg.inv(RotMnew2old)
    
    # --- Rotate 3D points ---
    old_3D = torch.stack([X_3D, Y_3D, Z_3D], dim=1).float()
    new_3D = (RotMold2new @ old_3D.T).T
    new_2D_x, new_2D_y = map3DTo2DERP_batch(new_3D[:,0], new_3D[:,1], new_3D[:,2], ERPimg_W, ERPimg_H)
    
    #new_2D_x = torch.clamp(new_2D_x - (ERPimg_W-ROI_W)/2, 0, ROI_W)
    #new_2D_y = torch.clamp(new_2D_y - (ERPimg_H-ROI_H)/2, 0, ROI_H)

    # --- Create ERP grid (vectorized) ---
    xs = torch.arange(0, ERPimg_W, device=device)
    ys = torch.arange(0, ERPimg_H, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    grid_x = grid_x.flatten().float()
    grid_y = grid_y.flatten().float()
    
    # --- Map ERP grid to 3D ---
    new_3D_x, new_3D_y, new_3D_z = map2DOfsetERPTo3D_batch(grid_x, grid_y, ERPimg_W, ERPimg_H, offset)
    new_3D_grid = torch.stack([new_3D_x, new_3D_y, new_3D_z], dim=1).float()
    
    # --- Rotate back and sample old ERP pixels ---
    old_3D_grid = (RotMnew2old @ new_3D_grid.T).T
    old_2D_x, old_2D_y = map3DTo2DERP_batch(old_3D_grid[:,0], old_3D_grid[:,1], old_3D_grid[:,2], ERPimg_W, ERPimg_H)
    old_2D_x = old_2D_x.clamp(0, ERPimg_W-1).long()
    old_2D_y = old_2D_y.clamp(0, ERPimg_H-1).long()
    sampled_pixels = old_rgb_erp[old_2D_y, old_2D_x]
    new_rgb_erp = sampled_pixels.view(ERPimg_H, ERPimg_W, 3).cpu().numpy().astype(np.uint8)
    
    
    masks_str_list = []
    n_entities = len(entities)
    # --- Draw all bounding boxes efficiently ---
    img_np = cv2.cvtColor(new_rgb_erp, cv2.COLOR_RGB2BGR)

    for idx in range(n_entities):
        entity = entities[idx]
        color, color_name = color_palette[idx % len(color_palette)]
        start, end = idx*4, (idx+1)*4
        x1_rel, y1_rel = int(new_2D_x[start:end].min()), int(new_2D_y[start:end].min())
        x2_rel, y2_rel = int(new_2D_x[start:end].max()), int(new_2D_y[start:end].max())
        cv2.rectangle(img_np, (x1_rel, y1_rel), (x2_rel, y2_rel), color, 30)
        masks_str_list.append(f"{entity}: {color_name} line")
    # --- Save final ERP ROI image ---
    cv2.imwrite(ROI_path, img_np)
    
    return masks_str_list

def generate_masks_prompt_v3(ERPimage_path,CMPimage_path, entities, boxes, mask_path,CMPimg_W,CMPimg_H,
                                ERPimg_W,ERPimg_H,face_W,face_H,device,ROI_gen_mode):
    if ROI_gen_mode == 1: #Draw bbox on CMP
        return generate_masks_prompt(CMPimage_path, entities, boxes, mask_path),False
    elif ROI_gen_mode == 2: #Rotation and using CMP Not yet coded
        return createCMPRoIfromCMP4x3BBoxes(entities,mask_path,corners_x,corners_y, 
                                         CMPimg_W,CMPimg_H, ERPimg_W,ERPimg_H,face_W, face_H, ERPimage_path,device),True
    elif ROI_gen_mode ==3: #Rotation and using ERP
        corners_x, corners_y, faceid = get_corner_coordinates_torch_v2(boxes, CMPimg_W, CMPimg_H, device)
        return createRoIfromCMP4x3BBoxes(entities,mask_path,corners_x,corners_y, 
                                         CMPimg_W,CMPimg_H, ERPimg_W,ERPimg_H,face_W, face_H, ERPimage_path,device),True
    else:
        raise ValueError("ROI_gen_mode must be 1,2, or 3!")
        