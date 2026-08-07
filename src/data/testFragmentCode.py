import xml.etree.ElementTree as ET

xml_content = """<annotation>
<size>
    <width>200</width>
    <height>200</height>
</size>
<object>
    <name>car</name>
    <bndbox>
        <xmin>60</xmin>
        <ymin>60</ymin>
        <xmax>140</xmax>
        <ymax>140</ymax>
    </bndbox>
</object>
<object>
    <name>car</name>
    <bndbox>
        <xmin>0</xmin>
        <ymin>0</ymin>
        <xmax>40</xmax>
        <ymax>40</ymax>
    </bndbox>
</object>
</annotation>"""



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

with open("test_annotation.xml", "w") as f:
    f.write(xml_content)
print(parse_voc_xml("test_annotation.xml", {"car":0}))
