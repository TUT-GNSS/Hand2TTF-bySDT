import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import UserDataset
import numpy as np
from xml.dom.minidom import Document
from models.model import SDT_Generator
import tqdm

def split_strokes_for_svg(coordinates):
    ids = np.where(coordinates[:, -1] == 1)[0]
    if len(ids) < 1:
        ids = np.where(coordinates[:, 3] == 1)[0] + 1
        if len(ids) < 1:
            ids = np.array([len(coordinates)])
            xys_split = np.split(coordinates, ids, axis=0)[:-1]
        else:
            xys_split = np.split(coordinates, ids, axis=0)
    else:
        remove_end = np.split(coordinates, ids, axis=0)[0]
        ids = np.where(remove_end[:, 3] == 1)[0] + 1
        xys_split = np.split(remove_end, ids, axis=0)
    return xys_split

def export_to_svg(points, save_path, canvas_size=1000, board=50, stroke_width=2):
    points = points.copy()
    points[:, 0] = np.cumsum(points[:, 0])
    points[:, 1] = np.cumsum(points[:, 1])
    xys_split = split_strokes_for_svg(points)
    min_x = min_y = 1e9
    max_x = max_y = -1e9
    for stroke in xys_split:
        if len(stroke) == 0:
            continue
        xs, ys = stroke[:, 0], stroke[:, 1]
        min_x = min(min_x, np.min(xs))
        max_x = max(max_x, np.max(xs))
        min_y = min(min_y, np.min(ys))
        max_y = max(max_y, np.max(ys))
    original_width = max_x - min_x
    original_height = max_y - min_y
    p_canvas = canvas_size - 2 * board
    p_canvas_w = p_canvas_h = p_canvas
    if original_height > original_width:
        scaleWidth = p_canvas_h / original_height * original_width
        scaleHeith = p_canvas_h
    else:
        scaleWidth = p_canvas_w
        scaleHeith = p_canvas_w / original_width * original_height
    svg_paths = []
    for stroke in xys_split:
        if len(stroke) == 0:
            continue
        xs, ys = stroke[:, 0], stroke[:, 1]
        xs_new = (xs - min_x) / original_width * scaleWidth + (p_canvas_w - scaleWidth) / 2 + board
        ys_new = (ys - min_y) / original_height * scaleHeith + (p_canvas_h - scaleHeith) / 2 + board
        path = f"M {xs_new[0]:.2f},{ys_new[0]:.2f} "
        for x, y in zip(xs_new[1:], ys_new[1:]):
            path += f"L {x:.2f},{y:.2f} "
        svg_paths.append(path.strip())
    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    svg.setAttribute('width', str(canvas_size))
    svg.setAttribute('height', str(canvas_size))
    doc.appendChild(svg)
    for path_str in svg_paths:
        path_elem = doc.createElement('path')
        path_elem.setAttribute('d', path_str)
        path_elem.setAttribute('stroke', 'black')
        path_elem.setAttribute('stroke-width', str(stroke_width))
        path_elem.setAttribute('fill', 'none')
        svg.appendChild(path_elem)
    with open(save_path, 'w', encoding='utf-8') as f:
        doc.writexml(f, addindent='  ', newl='\n', encoding='utf-8')

def main(opt):
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    test_dataset = UserDataset(cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, opt.style_path)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                             shuffle=True,
                                             sampler=None,
                                             drop_last=False,
                                             num_workers=cfg.DATA_LOADER.NUM_THREADS)
    os.makedirs(os.path.join(opt.save_dir, 'svg'), exist_ok=True)
    output_svg_dir = os.path.join(opt.save_dir, 'svg')
    model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
    if len(opt.pretrained_model) > 0:
        model_weight = torch.load(opt.pretrained_model)
        model.load_state_dict(model_weight)
        print('load pretrained model from {}'.format(opt.pretrained_model))
    else:
        raise IOError('input the correct checkpoint path')
    model.eval()
    batch_samples = len(test_loader)
    data_iter = iter(test_loader)
    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_samples)):
            data = next(data_iter)
            img_list, char_img, char = data['img_list'].cuda(), data['char_img'].cuda(), data['char']
            preds = model.inference(img_list, char_img, 120)
            bs = char_img.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)
            preds = preds.detach().cpu().numpy()
            for i, pred in enumerate(preds):
                this_char = char[i]
                unicode_hex = f"{ord(this_char):04x}"
                svg_path = os.path.join(output_svg_dir, f"u+{unicode_hex}.svg")
                export_to_svg(pred, save_path=svg_path)
                # print(f"SVG generated: {svg_path} for char: {this_char}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_USER.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated/ttf/Chinese_User', help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default='checkpoints/checkpoint-iter199999.pth', required=True, help='continue train model')
    parser.add_argument('--style_path', dest='style_path', default='style_samples', help='dir of style samples')
    opt = parser.parse_args()
    main(opt)