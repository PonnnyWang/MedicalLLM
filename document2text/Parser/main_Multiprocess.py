import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _parser import MultiFormatParser
import os
import tqdm
import json
from utils import save_to_txt, merge_txt_files
import argparse

TXT_PATHS_DIR = "output\paths"
FILE_ROOT = r"E:\CAS_PROJECT\MedLLM\MedicalLLM\text2txt\data"
SAVE_ROOT = r"output\txt_file"

def get_all_file_path(FILE_ROOT):
    '''
        获取所有文件路径
    '''
    file_paths = []
    for root, dirs, files in os.walk(FILE_ROOT):
        for file in files:
            if file.endswith(('.pdf', '.txt', '.docx', '.png', '.jpg')):
                file_paths.append(os.path.join(root, file))
    print("len(file_paths): {}".format(len(file_paths)))
    return file_paths

def write_path():
    '''
        将所有文件路径分别归入8个txt文件中，以便使用多线程加速处理
    '''
    file_paths = get_all_file_path(FILE_ROOT)
    file_paths = [file_paths[i::8] for i in range(8)]
    for i in range(8):
        if not os.path.exists(TXT_PATHS_DIR):
            os.makedirs(TXT_PATHS_DIR)
        with open(os.path.join(TXT_PATHS_DIR, str(i) + ".txt"), 'w') as f:
            for path in file_paths[i]:
                f.write(path + "\n")


def inference(mode):
    '''
        使用多进程将8个txt文件中的文件路径分别传入，进行处理
        保存单个文件转换后的单个txt文本，同时保存各个目录下合并后的txt文件
    '''
    write_path()
    assert mode in [0, 1, 2, 3, 4, 5, 6, 7]
    paths = []
    with open(os.path.join(TXT_PATHS_DIR, str(mode) + ".txt"), 'r', encoding='gbk') as f:
        for line in f:
            line = line.strip()
            file_extension = os.path.splitext(line)[1] 
            if file_extension in ['.pdf', '.txt', '.docx', '.png', '.jpg']:
                txt_path = line.replace(file_extension, ".txt").replace(FILE_ROOT, SAVE_ROOT)
            else:
                continue
            # txt_path = line.replace(FILE_ROOT, SAVE_ROOT)
            if os.path.exists(txt_path):
                continue
            paths.append(line)
    print("mode: {}, len(paths): {}".format(mode, len(paths)))

    parser = MultiFormatParser(split_length=500)    # 默认中文，分割长度100
    for file_path in tqdm.tqdm(paths):
        try:
            text = parser.extract_text(file_path)
            directory = os.path.dirname(file_path)
            # 获取单个转换后的txt文本
            output_dir = SAVE_ROOT + directory.replace(FILE_ROOT, "")
            save_to_txt(text, file_path, output_dir)
            # 获取合并后的txt文件
            merged_output_dir = SAVE_ROOT + '_merged'             
            output_filename = os.path.basename(directory) + '.txt'  # 获取目录名称作为输出文件名
            merge_txt_files(output_dir, merged_output_dir, output_filename)
        except Exception as e:
            with open("error.txt", 'a') as error_log:
                error_log.write(file_path + "\n")
                error_log.flush()

def main():
    '''
    单独加载不同mode进程
    '''
    parser = argparse.ArgumentParser(description="Run inference with different modes")
    parser.add_argument("mode", type=int, help="Mode to run (0-7)")
    args = parser.parse_args()
    mode = args.mode
    inference(mode)

if __name__ == "__main__":

    for i in range(8):
        inference(i)

    # main()

