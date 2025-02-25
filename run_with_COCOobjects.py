import os
import cv2
import torch
import deeplake
from groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np

# Create output directory
OUTPUT_FOLDER = "Complex Objects Annotated"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Check if Metal (MPS) is available for GPU acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load GroundingDINO model
MODEL_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT = "weights/groundingdino_swint_ogc.pth"
print(f"Loading model from {MODEL_CHECKPOINT}...")
model = load_model(MODEL_CONFIG, MODEL_CHECKPOINT)
model = model.to(device)
print("Model loaded successfully")

# Define standard COCO categories (80 classes)
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
COCO_CATEGORIES = set(category.lower() for category in COCO_CATEGORIES)
print(f"Loaded {len(COCO_CATEGORIES)} COCO classes.")

# Try to load the DeepLake dataset as fallback
try:
    print("Attempting to load DeepLake COCO dataset...")
    ds = deeplake.load("hub://activeloop/coco-test")
    print("DeepLake dataset loaded successfully")
    
    # Optionally explore dataset structure
    print("Available tensors:", list(ds.tensors.keys()))
    
    # Try to extract additional categories if present in the dataset
    # This is optional and depends on the dataset structure
    additional_categories = set()
    if 'categories' in ds.tensors.keys():
        # If categories tensor exists directly
        for category in ds.categories.data()['value']:
            additional_categories.add(category.lower())
    
    # Update COCO_CATEGORIES if we found more
    if additional_categories:
        print(f"Found {len(additional_categories)} additional categories in dataset")
        COCO_CATEGORIES.update(additional_categories)
        print(f"Updated to {len(COCO_CATEGORIES)} total categories")
except Exception as e:
    print(f"Note: DeepLake dataset could not be loaded: {e}")
    print("Continuing with standard COCO categories only")

# Function to extract objects from folder name without spaCy
def extract_objects_from_folder_name(folder_name):
    # Simple approach: Split by underscore and filter based on COCO categories
    words = folder_name.replace("_", " ").lower().split()
    
    # Check if any word is in COCO categories
    coco_words = [word for word in words if word in COCO_CATEGORIES]
    
    # Fallback to all words if no COCO objects found
    if not coco_words:
        print(f"No COCO objects found in '{folder_name}'. Using all words as fallback.")
        # Just use all words as potential objects
        return words
    
    return coco_words

# Define input folder
INPUT_FOLDER = "Complex Objects"
if not os.path.exists(INPUT_FOLDER):
    print(f"Error: Input folder '{INPUT_FOLDER}' does not exist")
    exit(1)

# Process input folders
print(f"Processing folders in {INPUT_FOLDER}...")
for prompt_folder in sorted(os.listdir(INPUT_FOLDER)):
    prompt_path = os.path.join(INPUT_FOLDER, prompt_folder)
    output_prompt_path = os.path.join(OUTPUT_FOLDER, prompt_folder)
    
    if not os.path.isdir(prompt_path):
        continue
        
    os.makedirs(output_prompt_path, exist_ok=True)
    
    # Extract objects from folder name
    text_prompts = extract_objects_from_folder_name(prompt_folder)
    
    # Skip folders without valid objects
    if not text_prompts:
        print(f"Skipping '{prompt_folder}' (no objects detected).")
        continue
        
    print(f"\nProcessing '{prompt_folder}' with objects: {text_prompts}")
    
    # Process each image in the folder
    for i in range(1, 6):
        image_filename = f"variation_{i}.png"
        image_path = os.path.join(prompt_path, image_filename)
        output_path = os.path.join(output_prompt_path, f"annotated_{i}.png")
        
        if not os.path.exists(image_path):
            print(f"  Skipping {image_filename} (file not found)")
            continue
            
        print(f"  Processing: {image_filename}")
        
        try:
            # Process each detected object separately for better results
            final_boxes = []
            final_logits = []
            final_phrases = []
            
            image_source, image = load_image(image_path)
            
            # Try each prompt separately for better detection
            for text_prompt in text_prompts:
                print(f"    Detecting: {text_prompt}")
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=text_prompt,
                    box_threshold=0.35,
                    text_threshold=0.25
                )
                
                if boxes is not None and len(boxes) > 0:
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    if isinstance(logits, torch.Tensor):
                        logits = logits.cpu().numpy()
                    
                    final_boxes.extend(boxes)
                    final_logits.extend(logits)
                    final_phrases.extend(phrases)
            
            # Convert to proper format if needed
            if final_boxes:
                final_boxes = np.array(final_boxes)
                final_logits = np.array(final_logits)
                
                # Convert back to torch tensors if needed by annotate function
                if isinstance(image_source, np.ndarray):
                    final_boxes = torch.from_numpy(final_boxes)
                    final_logits = torch.from_numpy(final_logits)
                
                # Annotate the image with detected objects
                annotated_frame = annotate(
                    image_source=image_source, 
                    boxes=final_boxes, 
                    logits=final_logits, 
                    phrases=final_phrases
                )
                
                # Save the annotated image
                success = cv2.imwrite(output_path, annotated_frame)
                if success:
                    print(f"    Saved: {os.path.basename(output_path)}")
                else:
                    print(f"    Failed to save: {os.path.basename(output_path)}")
            else:
                print(f"    No objects detected in {image_filename}")
        
        except Exception as e:
            print(f"    Error processing {image_filename}: {e}")

print("\nProcessing complete!")
print(f"Annotated images saved to '{OUTPUT_FOLDER}' directory")
