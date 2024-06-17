import os
import sys
sys.path.append('../') 

import tqdm
import json
from Parser.Utils import utils
from _parser import MultiFormatParser
import argparse
from concurrent.futures import ProcessPoolExecutor
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

TXT_PATHS_DIR = "output\paths"
FILE_ROOT = r"E:\CAS_PROJECT\MedLLM_local\data\PDF\test"
SAVE_ROOT = r"output\txt_file"
error_log = "output\error.txt"

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
        将所有文件路径分别归入8个txt文件中，以便使用多进程管理
    '''
    file_paths = get_all_file_path(FILE_ROOT)
    file_paths = [file_paths[i::8] for i in range(8)]
    for i in range(8):
        if not os.path.exists(TXT_PATHS_DIR):
            os.makedirs(TXT_PATHS_DIR)
        with open(os.path.join(TXT_PATHS_DIR, str(i) + ".txt"), 'w') as f:
            for path in file_paths[i]:
                f.write(path + "\n")


def process_file(file_path):
    '''
        处理单个文件
    '''
    try:
        parser = MultiFormatParser()
        text = parser.extract_text(file_path)
        directory = os.path.dirname(file_path)
        # 获取单个转换后的txt文本
        output_dir = SAVE_ROOT + directory.replace(FILE_ROOT, "")
        utils.save_to_txt(text, file_path, output_dir)
        # 获取合并后的txt文件
        merged_output_dir = SAVE_ROOT + '_merged'             
        output_filename = os.path.basename(directory) + '.txt'  
        utils.merge_txt_files(output_dir, merged_output_dir, output_filename)
    except Exception as e:
        with open(error_log, 'a') as error:
            error.write(f"{file_path} Error: {str(e)}\n")  


def process_mode_files(mode_path_file):
    '''
    处理单个模式中的txt文件
    '''
    mode_path = os.path.join(TXT_PATHS_DIR, mode_path_file)
    with open(mode_path, 'r', encoding='gbk') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    for file_path in file_paths:
        try:
            process_file(file_path)
        except Exception as e:
            with open(error_log, 'a') as error:
                error.write(f"{file_path} Error: {str(e)}\n")

def inference(mode):
    '''
        处理每个mode中的所有文件
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
            if os.path.exists(txt_path):
                continue
            paths.append(line)
    print("mode: {}, len(paths): {}".format(mode, len(paths)))

    for file_path in tqdm.tqdm(paths):
        try:
            process_file(file_path)
        except Exception as e:
            with open(error_log, 'a') as error:
                error.write(f"{file_path} Error: {str(e)}\n")
                error.flush()

def main(num_processes=8):
    '''
    使用多进程处理
    '''
    parser = argparse.ArgumentParser(description="Run inference in parallel across all modes using multiple processes.")
    parser.add_argument("--num_processes", type=int, default=num_processes, help="Number of processes to use.")
    args = parser.parse_args()

    write_path()
    parser = MultiFormatParser()

    # 创建一个总进度条
    total_files = sum(len(open(os.path.join(TXT_PATHS_DIR, str(mode) + ".txt"), 'r').readlines()) for mode in range(args.num_processes))
    total_progress_bar = tqdm.tqdm(total=total_files, desc="Total Progress")

    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = {executor.submit(process_mode_files, str(mode) + ".txt") for mode in range(args.num_processes)}    
        mode_file_counts = {mode: len(open(os.path.join(TXT_PATHS_DIR, str(mode) + ".txt"), 'r').readlines()) for mode in range(args.num_processes)}
        mode_progress_bars = {mode: tqdm.tqdm(total=count, desc=f"Mode {mode}") for mode, count in mode_file_counts.items()}
        for future in as_completed(futures):
            mode = int(future.args[0].split('.')[0])
            try:
                mode_progress_bars[mode].update(1) # 更新mode
                total_progress_bar.update(1) # 更新总进度
            except Exception as exc:
                print(f"An error occurred during processing: {exc}")

    # 确保所有进度条都关闭
    for mode in mode_progress_bars:
        mode_progress_bars[mode].close()
    total_progress_bar.close()


if __name__ == "__main__":

    for i in range(8):
        inference(i)

    # main()

