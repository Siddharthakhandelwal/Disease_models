from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("spine.pt")

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

Breast -
    Class Names :
        0: cancer
        1: normal

Hair -
    Class Names:
        0: Alopecia areata
        1: Head_Lice
        2: Psoriasis
        3: folliculitis

Pneumonia -
    Class Names:
        0: pneumonia negative
        1: pneumonia positive

Spine -
    Class Names:
        0: L1
        1: L2
        2: L3
        3: L4
        4: L5
        5: S1
'''