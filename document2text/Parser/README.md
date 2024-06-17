### 准备
#### 新建环境管理
`conda create --name medicalLLM python==3.10.8`
`conda activate medicalLLM`
#### 安装框架
1. **PaddlePadlle**
   * pip安装：
      CUDA11.8以上
        > python -m pip install paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple

   * conda安装：
       CUDA11.7以上
       > conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
   * 其它CUDA版本请查看：[PaddlePaddle安装](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html)
2. **layoutparser**
   * 快速安装：
        ```
        pip install layoutparser # Install the base layoutparser library with
        pip install "layoutparser[layoutmodels]" # Install DL layout model toolkit
        pip install "layoutparser[ocr]" # Install OCR toolkit
        ```
   * 了解细节：
      [GitHub仓库:Layout-Parser](https://github.com/Layout-Parser/)
3. **ppstructural**
   * 安装whl包
    ```
    pip install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
    ```
   * 了解细节：
      [GitHub仓库:PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppstructure)
#### 安装依赖包
`pip install -r requirements.txt`

### 运行
成功设置环境并对`FILE_ROOT`和`SAVE_ROOT`进行必要的调整后，运行该程序将首先扫描并收集位于目录FILE_ROOT中的所有 PDF 文件，将它们分布到八个独立的文本文件中。提高并行处理的效率。
* 默认代码使用inference函数单进程执行该过程。
```
    for i in range(8):
        inference(i)
```
* 使用多进程并发执行(计算资源充足)。
  ```
  main()
  ```


