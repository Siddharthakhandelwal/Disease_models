from gradio_client import Client, handle_file
import os

client = Client("Siddhartha-khandelwal/Disease_models")

image_path = "hair_image.jpg"  # ✅ Adjust this if needed
print("Exists:", os.path.exists(image_path))

result = client.predict(
    input_image=handle_file(image_path),
    api_name="/predict"
)
print(result)
