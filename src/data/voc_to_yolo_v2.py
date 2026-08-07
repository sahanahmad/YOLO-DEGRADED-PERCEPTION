import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import csv

PATCH_SIZE = 15

IMG_FOLDER = "data/raw/RTTS/JPEGImages"
ANN_FOLDER = "data/raw/RTTS/Annotations"
LABEL_FOLDER = "data/labels"
SEVERITY_CSV = "data/severity_scores.csv"
CLASS_MAP = {"car": 0, "bus": 1, "bicycle": 2, "motorbike": 3, "person": 4}

#COMPUTES MEAN DARK CHANNEL VALUE OF AN IMAGE
#PROXY FOR HAZE SEVERITY
def dark_channel(image_bgr: np.ndarray, patch_size:int=PATCH_SIZE)->float:
    min_channel = np.min(image_bgr, axis=2).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(patch_size,patch_size))
    dark = cv2.erode(min_channel,kernel)
    return float(np.mean(dark))

#PARSE VOC XML FILE, EXTRACT IMAGE SIZE AND COVERT
#BBOXES TO YOLO FORMAT
def parse_voc_xml(xml_path, class_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        #print(name, xmin,ymin,xmax,ymax)
        cx= (xmin + xmax)/2/img_w
        cy = (ymin + ymax)/2/img_h
        bw = (xmax-xmin)/img_w
        bh = (ymax-ymin)/img_h
        #print(cx,cy,bw,bh)
        class_id = class_map[name]
        boxes.append((class_id, cx, cy, bw, bh))
    return img_w,img_h, boxes

#WRITE YOLO FORMATTED BOUNDING BOXES
# TO TXT FILE FOR ONE IMAGE
def write_yolo_label(txt_path, boxes):
    with open(txt_path, 'w') as f:
        for class_id, cx, cy, bw, bh in boxes:
            line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            f.write(line)

#BASED ON HEAZY CONDITION PROVIDE 
#SEVERITY LABEL
def get_bin(score,p33, p66):
    if score <= p33:
        return "low"
    elif score <= p66:
        return "medium"
    else:
        return "high"

#CALL PARSE_VOC_XML AND WRITE_YOLO_LABEL 
#READ ALL IMAGES, COMPUTE DARK CHANNEL SCORE FOR EACH
#RETURN RESULTS AS A LIST
def img_process():
    folder = IMG_FOLDER
    files = sorted(os.listdir(folder))
    results = []
    for i, filename in enumerate(files): #hardcoded 5 for testing code, will remove
        path = os.path.join(folder,filename)
        img = cv2.imread(path)
        score = dark_channel(img)
        results.append((filename,score))
        if i%500 == 0:
            print(f"Image read progress: {i}/{len(files)}")
    return results

#CONVERT ALL VOC XML ANNOTATIONS TO YOLO .TXT 
#LABEL FILES
def write_all_labels(class_map):
    img_folder = IMG_FOLDER
    ann_folder = ANN_FOLDER
    label_folder = LABEL_FOLDER
    os.makedirs(label_folder, exist_ok= True)

    files = sorted(os.listdir(img_folder))
    for i,filename in enumerate(files): #hardcoded 5 for testing, will remove
        base, ext = os.path.splitext(filename)
        xml_path = os.path.join(ann_folder, base + ".xml")
        img_w, img_h, boxes = parse_voc_xml(xml_path=xml_path, class_map=class_map)

        txt_path = os.path.join(label_folder, base + ".txt")
        write_yolo_label(txt_path=txt_path,boxes=boxes)
        #print(filename, "->", len(boxes), "boxes")
        if i%500 == 0:
            print(f"Label Writing Progress {i}/{len(files)}")

#SCAN ALL XML ANNOTATIONS AND RETURN SET OF 
#UNIQUE CLASS LABEL
def get_all_class_names():
    all_names = set()
    files = sorted(os.listdir(ANN_FOLDER))

    for filename in files:
        xml_path = os.path.join(ANN_FOLDER,filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            all_names.add(name)
    return all_names


if __name__ == "__main__":
    class_names = get_all_class_names(ann_folder=ANN_FOLDER)
    missing = class_names - set(CLASS_MAP.keys())
    if missing:
        raise ValueError(f"Found class names not in CLASS_MAP:{missing}")
    write_all_labels(class_map=CLASS_MAP)
    scores = img_process()
    severities = [severity for _,severity in scores]
    p33 = np.percentile(severities, 33)
    p66 = np.percentile(severities,66)

    with open(SEVERITY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "severity", "bin"])
        for filename, severity in scores:
            bin_level = get_bin(severity, p33=p33, p66=p66)
            writer.writerow([filename, severity, bin_level])




