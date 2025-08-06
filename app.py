import gradio as gr
import torch
from PIL import Image,ImageDraw, ImageFont
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from ultralytics import YOLO
import os
import uuid
import json
# ------------------ Stage 1: CLIP Classification ------------------

model = None
processor = None

def load_models():
    global model, processor
    if model is None or processor is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

KNOWN_IMAGES = {
    "bone_fracture": "images/bone.jpg",
    "alzheimers": "images/alzaimers.jpg",
    "spine": "images/spine.jpg",
    "brain_tumor": "images/brain_tumor.jpg",
    "Pneumonia": "images/pneumonia.jpg",
    "Hair":"images/hair_image.jpg",
    "Breast":"images/breast.jpeg"
}

model, processor = load_models()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

known_embeddings = {
    label: get_image_embedding(path) for label, path in KNOWN_IMAGES.items()
}

# ------------------ Stage 2A: TensorFlow Alzheimer Model ------------------

alzheimer_model = None

def load_alzheimer_model():
    global alzheimer_model
    if alzheimer_model is None:
        try:
            alzheimer_model = load_model("alzheimers.h5")
        except Exception as e:
            print(f"Error loading Alzheimer's model: {e}")
            return None
    return alzheimer_model

def Dementia(img_path):
    model = load_alzheimer_model()
    if model is None:
        return Image.open(img_path), "❌ Error: Could not load Alzheimer's model"
    
    class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    img = keras_image.load_img(img_path, target_size=(250, 250))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    # predicted_label = class_labels[predicted_index]
    # confidence = prediction[0][predicted_index]

    if predicted_index == 0:
        result_text = "🧠 Diagnosis: Mild Dementia detected. This early stage of cognitive decline may affect memory and thinking skills slightly. It’s important to consult a neurologist for cognitive testing and to explore strategies or medications that may slow progression."
    elif predicted_index == 1:
        result_text= "🧠 Diagnosis: Moderate Dementia detected. This stage typically impacts daily living and communication more significantly. Please seek immediate guidance from a healthcare professional to manage symptoms and plan supportive care."
    elif predicted_index == 2:
        result_text="🧠 Good News: No signs of dementia detected. While this scan appears normal, ongoing checkups and healthy lifestyle choices are essential to maintain cognitive function."
    elif predicted_index == 3:
        result_text= "🧠 Diagnosis: Very Mild Dementia detected. Subtle signs of memory or focus changes may be present. Early attention to cognitive health can make a meaningful difference—consider a consultation with a neurologist or memory care specialist."

    return img, result_text

# ------------------ Stage 2B: YOLOv8 Torch Models ------------------

YOLO_MODELS = {
    "bone_fracture": "bone.pt",
    "spine": "spine.pt",
    "brain_tumor": "brain_tumor.pt",
    "Pneumonia": "Pneumonia.pt",
    "Hair":"Hair.pt",
    "Breast":"breast.pt"
}


def yolo_predict(image_path, model_path,best_label):
    
    try:
        model = YOLO(model_path)
        results = model(image_path, save=True, save_txt=False)

        output_dir = results[0].save_dir
        output_img_path = os.path.join(output_dir, os.path.basename(image_path))

        no_detections = results[0].boxes is None or len(results[0].boxes) == 0

        if model_path == "bone.pt" and no_detections:
            with Image.open(output_img_path) as img:
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                width, height = img.size

                # Draw box over entire image
                draw.rectangle([(0, 0), (width, height)], outline="red", width=5)

                # Try to load font
                try:
                    font = ImageFont.truetype("arial.ttf", 30)
                except:
                    font = ImageFont.load_default()

                draw.text((10, 10), "Not fractured", fill="red", font=font)
                output_image = img.copy()
                response = "No fracture detected in the X-ray. This result indicates healthy bone structure, but if symptoms persist, consult an orthopedic specialist."
        else:
            # There are detections
            with Image.open(output_img_path) as img:
                output_image = img.convert("RGB").copy()
                output_image = img.copy()
                predicted_class = int(results[0].boxes.cls[0].item())
                with open("diagnosis_messages.json", "r", encoding="utf-8") as f:
                    response_texts = json.load(f)
                    response = response_texts[best_label][str(predicted_class)]

        return output_image, response

    except Exception as e:
        print(f"Error in YOLO prediction: {e}")
        return Image.open(image_path), f"❌ Error: Could not process image with YOLO model ({str(e)})"
# ------------------ Unified Gradio Pipeline ------------------

def pipeline(input_image):
    if input_image is None:
        return None, "❌ Please upload an image"
    
    try:
        temp_input = f"temp_{uuid.uuid4().hex}.jpg"
        input_image.save(temp_input)

        # Stage 1: CLIP Classification
        input_embedding = get_image_embedding(temp_input)
        similarities = {
            label: cosine_similarity(input_embedding.cpu().numpy(), known_embed.cpu().numpy())[0][0]
            for label, known_embed in known_embeddings.items()
        }
        best_label = max(similarities, key=similarities.get)
        best_score = similarities[best_label]

        if best_score < 0.25:
            os.remove(temp_input)
            return input_image, "❌ Upload a Valid Medical Image (low similarity score)"

        # Stage 2: Model Inference
        try:
            if best_label == "alzheimers":
                output_img, result_text = Dementia(temp_input)
            elif best_label in YOLO_MODELS:
                model_path = YOLO_MODELS[best_label]
                output_img, result_text = yolo_predict(temp_input, model_path,best_label)
            else:
                result_text = f"❌ Unknown image type: {best_label}"
                output_img = input_image
        except Exception as e:
            print(f"Error in model inference: {e}")
            output_img = input_image
            result_text = f"❌ Error during model inference: {str(e)}"

        # Clean up temp file
        if os.path.exists(temp_input):
            os.remove(temp_input)
            
        return output_img, result_text
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return input_image, f"❌ Error processing image: {str(e)}"

# ------------------ Gradio UI ------------------

demo = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="pil", label=None, elem_id="custom-upload"),
    outputs=[
        gr.Image(type="pil", label="Model Output Image"),
        gr.Text(label="Diagnosis Result"),
    ],
    title="Let's examine your reports!",
    description="",
    css="""
    .gradio-container {
        min-height: 100vh;
        padding: 2rem;
        background: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }
    @media (min-width: 640px) {
        .gradio-container {
            background: linear-gradient(to right, #f6f7ec, #e6e7dc);
            padding-top: 6rem;
        }
    }
    h1 {
        text-align: center;
        font-size: 1.875rem; /* Equivalent to text-3xl */
        font-weight: 700; /* Equivalent to font-bold */
        color: black;
        margin-bottom: 1rem; /* Equivalent to mb-4 */
        margin-top: 0;
    }
    #custom-upload {
        width: 100%;
        max-width: 32rem; /* Equivalent to max-w-lg */
        margin-top: 2.5rem; /* Equivalent to mt-10 */
    }
    #custom-upload .upload-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 2px dashed #d1d5db; /* Light gray dashed border for default look */
        border-radius: 0.5rem;
        padding: 1.5rem;
        background: white;
        transition: border-color 0.2s;
    }
    #custom-upload .upload-box:hover {
        border-color: #9ca3af; /* Slightly darker on hover */
    }
    #custom-upload .upload-box input[type="file"] {
        display: none;
    }
    #custom-upload .upload-box label {
        font-size: 1rem;
        font-weight: 500;
        color: #374151; /* Gray-700 for text */
        background: #e5e7eb; /* Gray-200 for button background */
        border-radius: 0.375rem;
        padding: 0.75rem 2rem;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .output-image, .output-text {
        width: 100%;
        max-width: 32rem;
        margin-top: 1.5rem;
    }
    """,
)


if __name__ == "__main__":
    demo.queue()
    demo.launch()