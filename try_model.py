from PIL import Image
from ultralytics import YOLO
import os

def yolo_predict(image_path, model_path):
    try:
        model = YOLO(model_path)
        results = model(image_path, save=True, save_txt=False)

        output_dir = results[0].save_dir
        output_img_path = os.path.join(output_dir, os.path.basename(image_path))

        # Force save and reload as RGB
        with Image.open(output_img_path) as img:
            img = img.convert("RGB")
            output_image = img.copy()  # Ensures no file handle lock

        result_text = f"✅ YOLOv8 detection done using `{os.path.basename(model_path)}`\n"
        result_text += f"📸 Saved processed image to `{output_img_path}`"

        return output_image, result_text

    except Exception as e:
        print(f"Error in YOLO prediction: {e}")
        return Image.open(image_path), f"❌ Error: Could not process image with YOLO model ({str(e)})"

print(yolo_predict("testing_images\hair_image.jpg","Hair.pt"))