**基于项目 SDT 实现了从拍照手写图到 ttf，感谢大佬 dailenson 的贡献。**

SDT🔗：https://github.com/dailenson/SDT

## 拍照上传手写字体生成 TTF 字体全流程

🔨 Requirements

```
conda create -n sdt python=3.8 -y
conda activate sdt
# install all dependencies
conda env create -f environment.yml
```

1. **图片预处理与单字分割**  
   使用 `split_handwritten_chinese.py` 对手写字照片进行处理：
   - 自适应二值化、骨架化、加粗、闭操作。
   - 利用 EasyOCR 自动检测并分割出每个汉字，保存为单字图片。

如果识别效果不好可以将 processed_img.png 自己截图尝试，可通过调整代码中 stroke_width 调整字体粗细

```shell
python split_handwritten_chinese.py
```

2. **AI 推理生成点序列并导出 SVG**  
   使用 `user_generate_ttf.py`：
   - 加载`checkpoint-iter199999.pth`模型，对每个单字图片进行推理，输出点序列。
   - 直接将每个汉字的点序列转换为 SVG 矢量文件，SVG 文件以 Unicode 命名（如 `u+4e00.svg`），保存在 `Generated/ttf/Chinese_User/svg` 目录下。

提前下载 CHINESE_USER.yml 到 configs 和 checkpoint-iter199999.pth 到 checkpoint

```shell
python user_generate_ttf.py --cfg configs/CHINESE_USER.yml --dir Generated/ttf/Chinese_User --pretrained_model checkpoints/checkpoint-iter199999.pth --style_path style_samples
```

3. **SVG 批量转为 TTF 字体文件**  
   使用 `svg2ttf.py`（需安装 fontforge）：
   - 批量导入所有 SVG 文件。
   - 自动将线条“描边转面”，生成封闭的字形轮廓，适合字体格式。
   - 合成并导出最终的 TTF 字体文件（如 `HandwritingFont.ttf`）。

代码中可通过调整 stroke_width 调整生成字体粗细

```shell
fontforge --script svg2ttf.py
```
