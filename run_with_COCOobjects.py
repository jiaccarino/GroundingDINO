import fiftyone as fo
import fiftyone.zoo as foz
import spacy
import os
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Check if Metal (MPS) is available for GPU acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load GroundingDINO model (updated for Mac)
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
model = model.to(device)

# Define input/output folders
INPUT_FOLDER = "Complex Objects"
OUTPUT_FOLDER = "Complex Objects Annotated"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load COCO dataset using FiftyOne
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
)

# Extract valid COCO categories
COCO_CATEGORIES = set()
for sample in dataset:
    if "detections" in sample.ground_truth:
        for detection in sample.ground_truth.detections:
            COCO_CATEGORIES.add(detection.label.lower())  # Normalize to lowercase

print(f"Loaded {len(COCO_CATEGORIES)} COCO classes.")

# Function to extract objects from caption and filter by COCO dataset
def extract_objects_from_caption(nlp, folder_name):
    caption = folder_name.replace("_", " ")
    doc = nlp(caption)
    noun_list = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    
    # Filter only valid COCO objects
    coco_nouns = [noun for noun in noun_list if noun in COCO_CATEGORIES]

    return coco_nouns

# Process input folders
for prompt_folder in sorted(os.listdir(INPUT_FOLDER)):
    prompt_path = os.path.join(INPUT_FOLDER, prompt_folder)
    output_prompt_path = os.path.join(OUTPUT_FOLDER, prompt_folder)

    if not os.path.isdir(prompt_path):
        continue  

    os.makedirs(output_prompt_path, exist_ok=True)  

    TEXT_PROMPT = extract_objects_from_caption(spacy.load("en_core_web_sm"), prompt_folder)

    # Skip folders without valid COCO objects
    if not TEXT_PROMPT:
        print(f"Skipping '{prompt_folder}' (no COCO objects detected).")
        continue

    print(f"Processing '{prompt_folder}' with objects: {TEXT_PROMPT}")

    for i in range(1, 6):
        image_filename = f"variation_{i}.png"
        image_path = os.path.join(prompt_path, image_filename)
        output_path = os.path.join(output_prompt_path, f"annotated_{i}.png")

        if not os.path.exists(image_path):
            print(f"Skipping {image_path} (file not found)")
            continue  

        print(f"Processing: {image_path} with prompt: {TEXT_PROMPT[0]}")

        try:
            image_source, image = load_image(image_path)

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT[0],  # Use filtered prompt
                box_threshold=0.35,
                text_threshold=0.25
            )

            if boxes is None or len(boxes) == 0:
                print(f"No objects detected in {image_path}")
                continue

            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

            success = cv2.imwrite(output_path, annotated_frame)
            if success:
                print(f"Saved: {output_path}")
            else:
                print(f"Failed to save: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Launch FiftyOne App for Visualization
session = fo.launch_app(dataset)
