"""
解析图片及PDF类型文件：使用YOLOv8训练的版面分析模型，使用paddleOCR进行文字识别。
也可以调用ocrAgent中的GCVOCR或TesseractOCR进行文字识别。
"""
import cv2
from Parser.Utils import ocrAgent
from Parser.Utils import utils
from ultralytics import YOLO

class DocumentExtractor():
    def __init__(self, model_path, ocr_lang):
        self.model_path = model_path
        self.ocr_lang = ocr_lang

    def detect_and_ocr(self, images):   
        threshold = 35  # 设置字符数阈值
        ocr_agent = ocrAgent.PaddleocrAgent(languages=self.ocr_lang, use_gpu=True, use_angle_cls=True)
        model = YOLO(self.model_path)    # 版面分析模型
        ocr_results = []
        for image in images:
            results = model(image, save_txt=False)    
            # 提取类别为 'Text' 的边界框， cls 属性为 1 
            boxes = results[0].boxes
            text_boxes = boxes.xyxy[boxes.cls == 1]
            text_blocks = utils.sort_text_blocks(text_boxes, image.shape[1])        
            for text_block in text_blocks:
                x1, y1, x2, y2 = map(int, text_block)             
                image_segment = image[y1:y2, x1:x2]     # 从原图中裁剪文本区域
                text = ocr_agent.detect(image_segment) 
                if len(text.replace(" ", "")) >= threshold:  
                    ocr_results.append(text)               
        return ocr_results

    def extract_image(self, image_path):
        images = cv2.imread(image_path)
        return self.detect_and_ocr(images)

    def extract_pdf(self, pdf_path):
        images = utils.pdf_to_images(pdf_path)    
        return self.detect_and_ocr(images)

