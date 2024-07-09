import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz 
import cv2
import numpy as np
from PIL import Image
from Parser.Utils.ocrAgent import *
from Parser.Utils.utils import *
from ultralytics import YOLO
from pdf2image import convert_from_path
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def detect_and_ocr(pdf_path, model_path):

    images = pdf_to_images(pdf_path)
    threshold = 35  # 设置字符数阈值
    ocr_agent = PaddleocrAgent(languages="ch", use_gpu=True, use_angle_cls=True)
    model = YOLO(model_path)

    annotated_images = []
    ocr_results = []
    for image in images:
        results = model(image, save_txt=False)    
        # 提取类别为 'Text' 的边界框， cls 属性为 1 
        boxes = results[0].boxes
        text_boxes = boxes.xyxy[boxes.cls == 1]
        text_blocks = sort_text_blocks(text_boxes, image.shape[1])   
        modified_img = results[0].orig_img.copy()   # 副本图像
        
        box_id = 0         
        for text_block in text_blocks:
            x1, y1, x2, y2 = map(int, text_block)
            cv2.rectangle(modified_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在每个边界框旁边添加编号
            cv2.putText(modified_img, str(box_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            box_id += 1 
            # 从原图中裁剪文本区域
            image_segment = image[y1:y2, x1:x2]
            # OCR
            try:
                text = ocr_agent.detect(image_segment) 
                if len(text.replace(" ", "")) >= threshold:  
                    ocr_results.append(text)
            except Exception as e:
                print(f"Error during OCR for block {text_block}: {e}")
                

        #  布局结果
        annotated_image = Image.fromarray(cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB))
        annotated_images.append(annotated_image)  

    return ocr_results, annotated_images

if __name__ == "__main__":
    pdf_path = r"/mnt/workspace/MedicalLLM/document2text/test/70岁以上女性乳腺癌患者的临床病理特征及预后分析_张圣泽.pdf"
    model_path = r"/mnt/workspace/MedicalLLM/document2text/Parser/models/v8_layout.pt"

    ocr_results, annotated_images = detect_and_ocr(pdf_path, model_path)
    print(ocr_results)
    # 输出OCR结果
    for line in ocr_results:
        print(f"text: {line}")
        print(f"length: {len(line)}")
    
    # 保存布局分析结果
    save_folder = r"/mnt/workspace/MedicalLLM/document2text/test/output2"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for idx, annotated_image in enumerate(annotated_images):
        save_path = os.path.join(save_folder, f"page_{idx + 1}.jpg")
        annotated_image.save(save_path)


