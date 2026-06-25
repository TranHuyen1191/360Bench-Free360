TEMPLATE_PROMPTS = {
    "Final_answer":
        """You are given a question about a 360-degree image and a scene graph describing the entities in the image, their attributes, the spatial relations between them, and the spatial relations between them and the viewer. Your task is to answer the question by analyzing the scene graph. 

Follow these steps carefully:
1. Analyze the scene graph: Analyze the attribute edges and spatial relation edges to find nodes relevant to the question.
2. Answer: Select the best answer from the multiple-choice options (A, B, C, or D). If the scene graph does not provide enough information to answer the question, answer with "CANNOT ANSWER".

Return your response strictly in the following JSON format:
{{
  "analysis": "<Your reasoning about scene graph>",
  "answer": "<Only 'A', 'B', 'C', 'D' OR 'CANNOT ANSWER'>"
}}


Example:
Question: How many red cars are below you?
A. 1
B. 2
C. 3
D. 0
Scene graph: 
List of nodes: car_1, car_2, car_3, book_1, bottom_view, behind_view
Attribute relations:
    car_1-->The image shows a red car parked on a street.
    car_2-->The image shows a red car parked on a street.
    car_3-->The image shows a blue car parked on a street.
    book_1-->The image shows a book on the table.
    bottom_view-->bottom of the scene and below the viewer
    behind_view-->back of the scene and behind the viewer
Spatial relations: car_2-->in-->bottom_view, car_3-->in-->bottom_view, book_1-->in-->bottom_view, car_1-->in-->behind_view
Output:
{{
  analysis: The question asks about red cars below you. Based on the attribute relations, find nodes relevant to `red cars below you': car_1, car_2, and bottom_view. Based on the spatial relations, only car_2 is in bottom_view. Therefore, the best answer is: A.",
  "answer": "A"
}}

Example:
Question: What is the color of the chair?
A. Blue
B. Red
C. White
D. Black
Scene graph: 
List of nodes: book_1, chair_1, bottom_view, behind_view
Attribute relations: 
    book_1-->The image shows a book on the table.
    chair_1-->The image shows a red chair.
    bottom_view-->bottom of the scene and below the viewer
    behind_view-->back of the scene and behind the viewer
Spatial relations: book_1-->in-->bottom_view, chair_1-->in-->behind_view
Output:
{{
  analysis: The question asks about the color of the chair. Based on the attribute relations, find nodes relevant to `chair': chair_1. Therefore, the best answer is: B. Red",
  "answer": "B"
}}

Now, process the given input:
Question: {}
Scene graph: {}
Output:""",
    "Caption_Attributes":
        """You are given an image crop extracted from a 360-degree image. Your task is to provide a detailed description of the {} visible in the crop to help answer the question: {}. If any text is visible, transcribe it completely. Do not guess or describe anything that cannot be clearly seen.

Return your response strictly in the following JSON format:
    {{"Description": "<detailed description in natural language>","Text": [<List of texts>]}}

Output:""",
    "Caption_spatialrelation":
        """You are given a 360-degree image and a question about that image. In the image, two entities are highlighted using colored bounding boxes, each with an associated label and boundary color. Your task is to describe the spatial relationship between these entities from the observer's perspective, to help answer the question.

Follow these steps carefully:  
1. Use the boundary colors to locate each labeled entity in the image. If any bounding box is not visible or if the spatial relationship cannot be determined, return an empty string "".
2. Analyze the spatial relationship between the visible entities from the observer's perspective to help answer the question. Do not guess or describe anything that cannot be clearly seen.
3. Describe the spatial relationship in details to help answer the question USING the entity labels in the format: Label1 --> spatial relationship --> Label2.

Return your response strictly in the following JSON format:
    {{"analysis": "<Your reasoning based on bounding boxes>","output": "<Label1 --> spatial relationship --> Label2>"}}

Now, process the given input:
Question: {}  
Entity labels: {}
Boundary colors: {}
Output:""",
"objdetection":
    """You are given a 360-degree image and a question about that image. Your task is to determine whether any entity mentioned in the question is visibly present in the given image.

Follow these steps carefully:
1. Read the question, identify all relevant entities mentioned in the question.
2. Outline the position of all relevant entities and output their bounding box coordinates in the format [x1, y1, x2, y2] (top-left and bottom-right corners), along with the entity label

Return your response strictly in the following JSON format:
        {{"analysis": "<list of relevant entities>","coordinates": [{{"bbox_2d": [x1,y1,x2,y2],"label": <entity_label>}},{{"bbox_2d": [x1,y1,x2,y2],"label": <entity_label>}},...]}}
        
Now, process the given input:  
Question: {}  
Output:""",
"objdetectionAndAttribute":
    """You are given a 360-degree image and a question about that image. Your task is to determine whether any entity mentioned in the question is visibly present in the given image.

Follow these steps carefully:
1. Read the question, identify all relevant entities mentioned in the question.
2. Outline the position of all relevant entities and output their bounding box coordinates in the format [x1, y1, x2, y2] (top-left and bottom-right corners), along with the entity label and its attributes to help answer the question. If any text is visible, transcribe it completely as an attribute. Do not guess or describe anything that cannot be clearly seen.

Return your response strictly in the following JSON format:
        {{"analysis": "<list of relevant entities>","coordinates and attributes": [{{"bbox_2d": [x1,y1,x2,y2],"label": <entity_label>,"attributes": <detailed description in natural language>}},{{"bbox_2d": [x1,y1,x2,y2],"label": <entity_label>,"attributes": <detailed description in natural language>}},...]}}
        
Now, process the given input:  
Question: {}  
Output:"""}




