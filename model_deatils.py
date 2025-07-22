from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("bone.pt")

# Get the class names
class_names = model.names
print("Class Names:")
for idx, name in class_names.items():
    print(f"{idx}: {name}")

''' 
Brain Tumor - 
    Class Names:
        0: Glioma
        1: Meningioma
        2: No tumor
        3: Pituitary

Bone Fracture -
    Class Names :
        0: Avlusion Fracture
        1: Comminuted Fracture
        2: Compression-Crush Fracutre
        3: Fracture Dislocation
        4: GreenStick Fracture
        5: HairLine Fracture
        6: Impact Fracture
        7: Intra-articular fracture
        8: Fracture
        9: Oblique fracture
        10: spiral fracture

'''