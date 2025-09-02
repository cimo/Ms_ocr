import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np

import craft_utils
import imgproc
import file_utils

from craft import CRAFT

from collections import OrderedDict

def merge_boxes_by_distance(polys: list, max_dx: int, max_dy: int) -> list:
    """
    Unisce i poligoni vicini basandosi solo sulla distanza in X e Y.
    
    Args:
        polys: list di np.array(N,2), poligoni originali
        max_dx: distanza massima orizzontale per unire due box
        max_dy: distanza massima verticale per unire due box
    Returns:
        list di np.array poligoni uniti (rettangoli minimi)
    """
    if len(polys) == 0:
        return []

    # Converti poligoni in [x_min, y_min, x_max, y_max]
    rects = []
    for p in polys:
        x_min = np.min(p[:,0])
        y_min = np.min(p[:,1])
        x_max = np.max(p[:,0])
        y_max = np.max(p[:,1])
        rects.append([x_min, y_min, x_max, y_max])
    rects = np.array(rects)

    merged = []
    used = [False] * len(rects)

    for i in range(len(rects)):
        if used[i]:
            continue
        # inizia un nuovo gruppo
        group = [rects[i]]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if used[j]:
                    continue
                # distanza minima tra il box j e qualsiasi box del gruppo
                for b in group:
                    dx = max(0, max(b[0], rects[j][0]) - min(b[2], rects[j][2]))
                    dy = max(0, max(b[1], rects[j][1]) - min(b[3], rects[j][3]))
                    if dx <= max_dx and dy <= max_dy:
                        group.append(rects[j])
                        used[j] = True
                        changed = True
                        break
        # crea bounding box rettangolare del gruppo
        group = np.array(group)
        x0 = group[:,0].min()
        y0 = group[:,1].min()
        x1 = group[:,2].max()
        y1 = group[:,3].max()
        merged.append(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]))
    return merged


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='craft_mlt_25k.pth', type=str, help='pretrained model')

parser.add_argument('--text_threshold', default=0.2, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.1, type=float, help='text low-bound score')

#parser.add_argument('--text_threshold', default=0.4, type=float, help='text confidence threshold')
#parser.add_argument('--low_text', default=0.2, type=float, help='text low-bound score')

parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1920, type=int, help='image size for inference')

parser.add_argument('--mag_ratio', default=4.0, type=float, help='image magnification ratio')
#parser.add_argument('--mag_ratio', default=2.0, type=float, help='image magnification ratio')

parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='image', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

root_folder: str = os.path.dirname(os.path.abspath(__file__))

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = os.path.join(root_folder, "result")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        merged_polys = merge_boxes_by_distance(polys, max_dx=0.1, max_dy=0.1)
        #merged_polys = merge_boxes_by_distance(polys, max_dx=3, max_dy=0.1)

        file_utils.saveResult(image_path, image[:,:,::-1], merged_polys, dirname=result_folder)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(result_folder, f"res_{filename}_mask.jpg")
        cv2.imwrite(mask_file, score_text)

    print("elapsed time : {}s".format(time.time() - t))