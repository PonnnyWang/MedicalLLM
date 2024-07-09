import os
import sys
import fitz
from PIL import Image
import json
import logging
import re
import zipfile
import numpy as np
import cv2

def split_long_text(text, split_length):
    '''
        按指定长度截断长文本
    '''
    lines = []
    text = text.replace('\n', ' ').replace('\r', '').replace('\f', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fa5,!?;:，。？!；：“”‘’（）《》%]', '', text)
    if split_length is None:
        return [text]        
    while len(text) > split_length:
        # 直接按指定长度截断，不考虑标点符号
        split_pos = split_length
        lines.append(text[:split_pos].rstrip())
        text = text[split_pos:].lstrip()
    lines.append(text.rstrip())
    return lines

def pdf_to_images(pdf_path):
    '''
    pdf转换为png，dpi调整识别图像分辨率，越高的分辨率ocr效果更好
    '''
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=500)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = np.array(img)
        images.append(img)
    return images

def sort_text_blocks(text_boxes, image_width):
    '''
    对PDF文本区域按顺序排序
    '''   
    text_blocks = text_boxes.tolist()   # Tensor2list

    left_blocks = []
    right_blocks = []

    for block in text_blocks:
        x_center = (block[0] + block[2]) / 2  # block:[x1, y1, x2, y2]
        if x_center < image_width / 2:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    left_blocks_sorted = sorted(left_blocks, key=lambda b: b[1])
    right_blocks_sorted = sorted(right_blocks, key=lambda b: b[1])

    return left_blocks_sorted + right_blocks_sorted

def save_to_txt(text, file_path, output_path):
    """
    将原始文件转换为txt文件，并保存到指定路径。
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_path, f"{file_name}.txt")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as txt_file:
            for index, line in enumerate(text, start=1):
                # txt_file.write(json.dumps({"text_line": text}, ensure_ascii=False) + '\n') # json格式
                txt_file.write(line + '\n') # 文本格式
        logging.info(f"Text has successfully translate to {file_name}.txt")
    except Exception as e:
        logging.error(f"Error saving text {file_name}.txt: {e}")

def merge_txt_files(source_dir, output_dir, output_filename):
    '''
    合并指定目录下的所有.txt文件到一个文件中，并保存到指定的输出目录。
    :param source_dir: 包含.txt文件的源目录路径。
    :param output_dir: 合并后文件要保存的目标目录路径。
    :param output_filename: 合并后文件的名称。
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 读取source_dir下的所有.txt文件，并将内容追加到output_dir中的一个文件中
    txt_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.txt')]
    with open(os.path.join(output_dir, output_filename), 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write('')       
                
def unzip_file(zip_file_path):
    # 解压zip文件：处理中文压缩包时会乱码
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('data/')
    print("unzip done!")

if __name__ == "__main__":
    pdf_to_images()
    # zip_file_path = r'data\zip_file.zip'
    # unzip_file(zip_file_path)

