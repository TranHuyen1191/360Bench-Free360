views_list = ["Front", "Behind", "Top", "Bottom", "Left", "Right"]
view_idx = ["front_view", "behind_view", "top_view", "bottom_view", "left_view", "right_view"]

att_view = {
    "front_view": "front of the scene and front of the viewer",
    "behind_view": "back of the scene and behind the viewer",
    "top_view": "top of the scene and above the viewer",
    "bottom_view": "bottom of the scene and below the viewer",
    "left_view": "left side of the scene and to the left of the viewer",
    "right_view": "right side of the scene and to the right of the viewer"
}

# Full list of relations between all views
full_view_relations = [
    "front_view-->opposite-->behind_view", "front_view-->below-->top_view", "front_view-->above-->bottom_view", "front_view-->right of-->left_view",
    "front_view-->left of-->right_view", "behind_view-->opposite-->front_view", "behind_view-->below-->top_view", "behind_view-->above-->bottom_view",
    "behind_view-->left of-->left_view", "behind_view-->right of-->right_view", "top_view-->above-->front_view", "top_view-->above-->behind_view",
    "top_view-->opposite-->bottom_view", "top_view-->above-->left_view", "top_view-->above-->right_view", "bottom_view-->below-->front_view",
    "bottom_view-->below-->behind_view", "bottom_view-->opposite-->top_view", "bottom_view-->below-->left_view", "bottom_view-->below-->right_view",
    "left_view-->left of-->front_view", "left_view-->right of-->behind_view", "left_view-->below-->top_view", "left_view-->above-->bottom_view",
    "left_view-->opposite-->right_view", "right_view-->right of-->front_view", "right_view-->left of-->behind_view", "right_view-->below-->top_view",
    "right_view-->above-->bottom_view", "right_view-->opposite-->left_view"
]

short_view_relations = [
    "front_view-->opposite-->behind_view", "front_view-->below-->top_view", "front_view-->above-->bottom_view", "front_view-->right of-->left_view",
    "front_view-->left of-->right_view", "behind_view-->below-->top_view", "behind_view-->above-->bottom_view", "behind_view-->left of-->left_view",
    "behind_view-->right of-->right_view", "top_view-->opposite-->bottom_view", "top_view-->above-->left_view", "top_view-->above-->right_view",
    "bottom_view-->below-->left_view", "bottom_view-->below-->right_view", "left_view-->opposite-->right_view"
]  # no overlap

VIEW_RELATIONS_LIST = {
    "short": short_view_relations,
    "full": full_view_relations,
}


def filter_relations(view_list, all_relations):
    """Return only the relations that occur between the given views."""
    filtered = []
    for rel in all_relations:
        parts = rel.split("-->")
        if len(parts) == 3:
            src, relation, tgt = parts
            # keep only if both source and target are in view_list
            if src in view_list and tgt in view_list:
                filtered.append(rel)

    return [att_view[view] for view in view_list], filtered

