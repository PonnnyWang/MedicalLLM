

import fitz 
import cv2
import layoutparser as lp
import numpy as np
from PIL import Image
from Parser.Utils.ocrAgent import *
from Parser.Utils.utils import *

def extract_text_from_images(images, model_path=None):
    ocr_results = []
    threshold = 35  # 设置字符数阈值
    ocr_agent = PaddleocrAgent(languages="ch", use_gpu=True, use_angle_cls=True)
    # ocr_agent = lp.TesseractAgent(languages="chi_sim + eng")
    # 加载模型
    model = lp.PaddleDetectionLayoutModel(
                                        #   model_path=model_path, 
                                          config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",  
                                          threshold=0.5,
                                          label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                                          enforce_cpu=False,
                                          enable_mkldnn=True)
    # model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',   
    #                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    #                                     label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    annotated_images = []
    for image in images:
        # BGR转RGB
        image = image[..., ::-1]
        # 检测
        layout = model.detect(image)
        # 过滤出文本和Title区域
        text_blocks = lp.Layout([b for b in layout if b.type == 'Text'])
        figure_blocks = lp.Layout([b for b in layout if b.type == 'Figure'])
        # 因为在图像区域内可能检测到文本区域，所以只需要删除它们
        text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

        text_blocks = sort_text_blocks(text_blocks, image.shape[1])

        # 对每个文本区域进行OCR
        for text_block in text_blocks:
            x1, y1, x2, y2 = map(int, text_block.coordinates)  # 获取文本块的坐标
            image_segment = image[y1:y2, x1:x2]  # 使用OpenCV裁剪图像
            try:
                text = ocr_agent.detect(image_segment)
                if len(text.replace(" ", "")) >= threshold:  # 检查文本字符数（不包括空格）
                    ocr_results.append(text)
            except Exception as e:
                print(f"Error during OCR for block {text_block}: {e}")

        # # 布局结果
        annotated_image = lp.draw_box(image, text_blocks, box_width=5, show_element_id=True)
        annotated_images.append(annotated_image)

    return ocr_results, annotated_images

if __name__ == "__main__":
    pdf_path = r"E:\CAS_PROJECT\MedLLM_local\data\PDF\test\闭链测定负荷对健康成人下肢位置觉的影响.pdf"

    images = pdf_to_images(pdf_path)
    ocr_results, annotated_images = extract_text_from_images(images)
    
    # 输出OCR结果
    for line in ocr_results:
        print(f"text: {line}")
        print(f"length: {len(line)}")
    
    # 保存布局分析结果
    save_folder = "E:\CAS_PROJECT\MedLLM_git\MedicalLLM\document2text\test\output"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for idx, annotated_image in enumerate(annotated_images):
        save_path = os.path.join(save_folder, f"page_{idx + 1}.jpg")
        annotated_image.save(save_path)


