import fontforge
import os

import time

def svg_folder_to_ttf(svg_dir, ttf_path, stroke_width=40):
    start_time = time.time()
    font = fontforge.font()
    font.encoding = 'UnicodeFull'
    svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]
    svg_files.sort()
    for svg_file in svg_files:
        # 从文件名提取unicode
        unicode_match = svg_file.lower().replace('.svg', '').replace('u+', '')
        codepoint = int(unicode_match, 16)
        glyph = font.createChar(codepoint)
        glyph.importOutlines(os.path.join(svg_dir, svg_file))
        # 关键：将线条转为封闭面
        glyph.stroke("circular", stroke_width, "round", "round")
        glyph.removeOverlap()
        glyph.simplify()
        glyph.width = 1000
        # print(f"Added {svg_file} as U+{codepoint:04X}")
    font.generate(ttf_path)
    print(f"TTF saved to {ttf_path}")
    end_time = time.time()
    print(f"耗时: {end_time - start_time:.2f} 秒")
if __name__ == "__main__":
    svg_folder_to_ttf('./Generated/ttf/Chinese_User/svg', './Generated/ttf/Chinese_User/HandwritingFont.ttf', stroke_width=20)