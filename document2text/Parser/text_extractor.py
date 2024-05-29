"""
解析图片及PDF类型文件：使用layerparser进行布局分析，使用paddleOCR进行文字识别。
也可以调用ocrAgent中的GCVOCR或TesseractOCR进行文字识别。
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from docx import Document as DocxDocument
from PIL import Image
import layoutparser as lp
import logging
from Utils.ocrAgent import PaddleocrAgent
from Utils.utils import split_long_text, pdf_to_images, save_to_txt, sort_text_blocks
            
class DocumentExtractor():
    def __init__(self, ocr_lang, split_length, model_path=None):
        '''
        默认从config中获得模型路径，也可以训练模型手动指定
        '''
        self.ocr_lang = ocr_lang
        self.model_path = model_path
        self.split_length = split_length

    def preprocess_image(self, image):
        # BGR转RGB
        image = image[..., ::-1]
        return image

    def detect_and_ocr(self, image):
        ocr_agent = PaddleocrAgent(languages=self.ocr_lang, use_gpu=True, use_angle_cls=True)
        model = lp.PaddleDetectionLayoutModel(model_path=self.model_path, 
                                              config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                              threshold=0.5,
                                              label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                                              enforce_cpu=False,
                                              enable_mkldnn=True)
        # 布局分析, 提取文本块
        layout = model.detect(image)
        text_blocks = lp.Layout([b for b in layout if b.type == 'Text'])
        figure_blocks = lp.Layout([b for b in layout if b.type == 'Figure'])
        text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
        text_blocks = sort_text_blocks(text_blocks, image.shape[1])
        
        ocr_results = []
        for text_block in text_blocks:
            x1, y1, x2, y2 = map(int, text_block.coordinates)
            image_segment = image[y1:y2, x1:x2]
            text = ocr_agent.detect(image_segment)
            if len(text.replace(" ", "")) >= 40:  
                ocr_results.append(text)      
        return ocr_results

    def extract_text(self, ocr_results):
        full_text = "".join(ocr_results)
        split_text = split_long_text(full_text, self.split_length)
        return split_text

    def extract_image(self, image_path):
        images = cv2.imread(image_path)
        ocr_results = []
        for image in images:
            image = self.preprocess_image(image)
            ocr_results.extend(self.detect_and_ocr(image))
        return self.extract_text(ocr_results)

    def extract_pdf(self, pdf_path):
        images = pdf_to_images(pdf_path)
        ocr_results = []
        for image in images:
            image = self.preprocess_image(image)
            ocr_results.extend(self.detect_and_ocr(image))
        return self.extract_text(ocr_results)
