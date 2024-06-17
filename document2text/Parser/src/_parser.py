'''
此文件用于解析txt\docx\PNG\JPG\PDF格式文件，并使用字符分割法将给定文件以自定义长度进行分割。
图片和PDF解析包括布局解析和文本提取两部分。
'''

import fitz
from docx import Document as DocxDocument
from text_extractor import DocumentExtractor
from Parser.Utils import utils
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

class MultiFormatParser():
    '''
        多格式解析器
        split_length是自定义字符分割长度，默认为None，表示不进行分割
    '''
    def __init__(self, ocr_lang='ch', use_gpu=True, split_length=None):
        self.ocr_lang = ocr_lang
        self.use_gpu = use_gpu
        self.split_length = split_length

    def extract_text(self, file_path):
        '''
            基于文件类型自动抽取
        '''
        file_type = file_path.lower().split('.')[-1]
        try:
            if file_type == 'pdf':
                return self.extract_pdf(file_path)
            elif file_type == 'docx':
                return self.extract_docx(file_path)
            elif file_type == 'txt':
                return self.extract_txt(file_path)
            elif file_type == 'png' or 'jpg':
                return self.extract_image(file_path)
            else:
                raise ValueError("Unsupported file type")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return []
 
    def extract_docx(self, docx_path):

        try:
            doc = DocxDocument(docx_path)  
            text_lines = []  
            full_text = ' '.join(para.text for para in doc.paragraphs)  # 防止分段
            split_text = utils.split_long_text(full_text, self.split_length) 
            text_lines.extend(split_text)  
            return text_lines
        except Exception as e:
            logging.error(f"Error extracting text from DOCX {docx_path}: {e}")
            return []

    def extract_txt(self, txt_path):

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                full_text = f.read()    # 防止分段
                split_text = utils.split_long_text(full_text, self.split_length)
                return split_text
        except Exception as e:
            logging.error(f"Error extracting text from TXT {txt_path}: {e}")
            return []

    def extract_image(self, image_path):
        extractor = DocumentExtractor(ocr_lang=self.ocr_lang, split_length=self.split_length, use_gpu=self.use_gpu)
        split_text = extractor.extract_image(image_path)
        return split_text

    def extract_pdf(self, pdf_path):
        extractor = DocumentExtractor(ocr_lang=self.ocr_lang, split_length=self.split_length, use_gpu=self.use_gpu)
        split_text = extractor.extract_pdf(pdf_path)
        return split_text        

if __name__ == "__main__":

    
    file_path = r"data\test\Williams综合征儿童的认知语言特点及康复启示.pdf"
    model_path = r"Parser\models\ppyolov2_r50vd_dcn_365e_publaynet_infer"
    output_path = r"output"
    parser = MultiFormatParser(ocr_lang='ch', use_gpu=True, split_length=None) # 使用CPU时，默认mkldnn加速
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 提取文本
    text = parser.extract_text(file_path)
    # save_to_txt(text, file_path, output_path)
    print(text)
    # 输出测试
    for line in text:
        print(f"text: {line}")
        print(f"length: {len(line)}")


