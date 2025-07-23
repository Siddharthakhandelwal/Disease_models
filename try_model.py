from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO
import os
import json
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

print(yolo_predict("testing_images/breast_cancer_positive.jpg","breast.pt","Breast"))