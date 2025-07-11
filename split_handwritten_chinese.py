import cv2
import os
import numpy as np
from skimage.morphology import skeletonize
import easyocr


input_image_path = 'shufa.jpg'
output_thin_dir = 'output_chars'
if not os.path.exists(output_thin_dir):
    os.makedirs(output_thin_dir)

# 1. 读取图片并二值化
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 35, 10
)
# 反色，确保字是白色，背景是黑色
if np.mean(binary) > 127:
    binary = 255 - binary

# 2. 整体骨架化
bool_img = (binary == 255)
skeleton = skeletonize(bool_img)
thin_img = (skeleton.astype(np.uint8)) * 255

# 3. 均匀加粗
stroke_width = 3 
circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width, stroke_width))
uniform_strokes = cv2.dilate(thin_img, circle_kernel, iterations=1)

# 4. 闭操作
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(uniform_strokes, cv2.MORPH_CLOSE, kernel3, iterations=1)

# 5. 反色，得到黑字白底
processed_img = 255 - closed

cv2.imwrite('processed_img.png', processed_img)
print("全局骨架化+均匀加粗+闭操作处理后图片已保存为 processed_img.png")

# 6. 用EasyOCR检测汉字区域（基于处理后的图片）
reader = easyocr.Reader(['ch_sim'])
results = reader.readtext(processed_img, detail=1)
char_count = 0

for bbox, text, conf in results:
    text = text.strip()
    pts = np.array(bbox).astype(int)
    x1 = np.min(pts[:, 0])
    y1 = np.min(pts[:, 1])
    x2 = np.max(pts[:, 0])
    y2 = np.max(pts[:, 1])
    char_img = processed_img[y1:y2, x1:x2]
    # 只处理较大区域
    if char_img.shape[0] > 20 and char_img.shape[1] > 20:
        # 如果是单字，直接处理
        if len(text) == 1 and '\u4e00' <= text <= '\u9fff':
            # 为分割出来的图片四周加padding并居中
            pad = 40  # 可根据需要调整留白多少
            char_img_padded = cv2.copyMakeBorder(char_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            char_count += 1
            filename = f"{char_count}.png"
            cv2.imwrite(os.path.join(output_thin_dir, filename), char_img_padded)
            print(f"保存: {filename}, 区域: ({x1},{y1})-({x2},{y2}), 识别为: {text}")
        else:
            # 对该区域再用EasyOCR识别
            sub_results = reader.readtext(char_img, detail=1)
            for sub_bbox, sub_text, sub_conf in sub_results:
                sub_text = sub_text.strip()
                if len(sub_text) == 1 and '\u4e00' <= sub_text <= '\u9fff':
                    sub_pts = np.array(sub_bbox).astype(int)
                    sx1 = np.min(sub_pts[:, 0])
                    sy1 = np.min(sub_pts[:, 1])
                    sx2 = np.max(sub_pts[:, 0])
                    sy2 = np.max(sub_pts[:, 1])
                    sub_img = char_img[sy1:sy2, sx1:sx2]
                    if sub_img.shape[0] > 20 and sub_img.shape[1] > 20:
                        # 为分割出来的图片四周加padding并居中
                        pad = 40  # 可根据需要调整留白多少
                        sub_img_padded = cv2.copyMakeBorder(sub_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
                        char_count += 1
                        sub_filename = f"{char_count}.png"
                        cv2.imwrite(os.path.join(output_thin_dir, sub_filename), sub_img_padded)
                        print(f"二次分割保存: {sub_filename}, 区域: ({sx1},{sy1})-({sx2},{sy2}), 识别为: {sub_text}")

print(f"共保存 {char_count} 个汉字到 {output_thin_dir} 文件夹。")
