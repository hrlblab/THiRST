"""Export the footprint polygon of every ST unit (Visium V2 spot circle or Visium HD bin square).

THiRST stage 0 - data preparation.

Same cropping as ``export_st_centers.py``; each in-tissue unit is written as a polygon in the 4x frame:
Visium V2 spots as 100-vertex circles (radius 28 px ~ 55 um diameter), Visium HD 8 um bins as squares
(+-4 px). These polygons are used for all IoU computations.

Outputs: ``<contour_root>/<case>/image_<section>_<x0>_<y0>_spot.csv``
           columns barcode, now_x_<k>, now_y_<k> (k = 0..99 for V2, 0..3 for HD)

Usage
-----
    python scripts/0_preprocessing/export_st_contours.py --config configs/example.yaml

Original script(s): draw_sep_svs_point_contour_fast_csv.py
"""
import sys
import os
import glob
import json

import numpy as np
import openslide
import pandas as pd
import cv2
from matplotlib.path import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import is_circular_spot, parse_args, read_in_tissue_positions, st_section  # noqa: E402


def generate_circle_contour(cx, cy, radius, num_points=100):
    # Generate angles
    angles = np.linspace(0, 2 * np.pi, num_points)

    # Calculate the x and y coordinates of the contour points
    contour = np.array([
            [int(cx + radius * np.cos(angle)), int(cy + radius * np.sin(angle))]
            for angle in angles
    ])

    # Reshape the contour to fit the format used by cv2.drawContours
    contour = contour.reshape((-1, 1, 2))

    return contour


def draw_line(case_name, IMG, JSON, output_folder, now_lv, num, cfg):
    output_dir = os.path.join(output_folder,case_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    svs = openslide.OpenSlide(IMG)


    with open(JSON, 'r') as f:
        annotation = json.load(f)

    for ai in range(len(annotation['features'])):
        # contours.append((annotation['features'][ai]['geometry']['coordinates']))
        try:
            now_ant = annotation['features'][ai]['properties']['classification']['name']
            if now_ant == cfg['region_class']:
                contours = annotation['features'][ai]['geometry']['coordinates']
        except:
            print('empty annotation')

    try:
        ann = np.array(contours)
    except:
        ann = np.array(contours[0])

        if len(ann.shape) == 2:
            ann = np.expand_dims(ann, axis = 0)

    x_min = ann[0,:,0].min()
    y_min = ann[0,:,1].min()
    x_max = ann[0,:,0].max()
    y_max = ann[0,:,1].max()

    cimg_root = os.path.join(output_dir, 'image_%s.png' % (os.path.basename(JSON).split('-')[-1].replace('.svs.geojson','')))

    a = os.path.basename(JSON).split('-')[-1].replace('.svs.geojson','')
    background_pixel = np.array(svs.read_region((int(np.round(x_min)), int(np.round(y_min))), 2, (10, 10)))[:, :, :3]

    padding = 1024

    # Space Ranger output of this section (configured under 'st_sections' in the YAML file).
    # 'positions' gives the spot/bin centres in the full-resolution WSI frame; 'in_tissue_positions'
    # (optional, defaults to 'positions') provides the in_tissue flag.
    st_cfg = st_section(cfg, a)
    circle = is_circular_spot(cfg, a)

    # if circle == 1:
    # 	return

    # if a!= '10':
    # 	return

    x_length = int(np.round((x_max - x_min)/4))
    y_length = int(np.round((y_max - y_min)/4))
    cimg = np.array(svs.read_region((int(np.round(x_min)),int(np.round(y_min))),2,(x_length, y_length)))[:,:,:3]

    mask = np.zeros_like(cimg)

    now_ann = ann.copy()
    now_ann[:, :, 0] = now_ann[:, :, 0] - x_min
    now_ann[:, :, 1] = now_ann[:, :, 1] - y_min

    cv2.fillPoly(mask, [(now_ann / 4).astype(int)], (255, 255, 255))
    masked_image = cv2.bitwise_and(cimg, mask)

    background_color = (np.mean(background_pixel[:, :, 0]), np.mean(background_pixel[:, :, 1]), np.mean(background_pixel[:, :, 2]))
    image_with_background = np.full_like(cimg, background_color)
    image_with_background[mask == 255] = cimg[mask == 255]


    extended_img = np.ones((y_length+padding * 2, x_length+padding * 2, 3))
    extended_img = np.full_like(extended_img, background_color) / 255.


    extended_img[padding:-padding:,padding:-padding:,:] = image_with_background/255.
    real_x = x_min - padding * 4
    real_y = y_min - padding * 4

    # plt.imsave(cimg_root.replace('.png', '_%d_%d.png' % (real_x, real_y)), extended_img)

    color_table = pd.DataFrame(columns=['barcode','x','y','R','G','B'])
    extended_img = np.zeros(extended_img.shape)
    cnt = 0

    # center = pd.DataFrame(columns = ['barcode','abs_x','abs_y','now_x','now_y'])

    position_intissue = read_in_tissue_positions(st_cfg)  # in_tissue flag matched by barcode

    point_array = np.zeros((len(position_intissue),2))

    point_array[:,1] = np.array(position_intissue['pxl_row_in_fullres'].tolist())
    point_array[:,0] = np.array(position_intissue['pxl_col_in_fullres'].tolist())

    # point_array[:,0] = np.array(position_intissue['pxl_row_in_fullres'].tolist())
    # point_array[:,1] = np.array(position_intissue['pxl_col_in_fullres'].tolist())

    polygon = Path(np.array(ann[0]))
    inside = polygon.contains_points(point_array)

    position_intissue_onsample = position_intissue[inside==1]

    point_array_onsample = point_array[inside==1]

    # center['barcode'] = position_intissue_onsample['barcode']
    # center['abs_x'] = (point_array_onsample[:,1]).astype(int).tolist()
    # center['abs_y'] = (point_array_onsample[:,0]).astype(int).tolist()
    # center['now_x'] = ((point_array_onsample[:,1] - real_y)/4).astype(int).tolist()
    # center['now_y'] = ((point_array_onsample[:,0] - real_x)/4).astype(int).tolist()

    radius = 28
    box_length = 4
    num_points = 100
    if circle:
        single_circle_contour = generate_circle_contour(0, 0, radius, num_points=num_points)
        batch_circle_contour = np.zeros((len(point_array_onsample),100, 1, 2))
        batch_circle_contour[:] = single_circle_contour
        batch_circle_contour[:,:,0,0] = batch_circle_contour[:,:,0,0] + np.round(((point_array_onsample[:,0] - real_x)/4)).astype(int)[:, np.newaxis]
        batch_circle_contour[:,:,0,1] = batch_circle_contour[:,:,0,1] + np.round(((point_array_onsample[:,1] - real_y)/4)).astype(int)[:, np.newaxis]

        columns = ['barcode'] + \
                  ['now_x_%d' % ki for ki in range(num_points)] + \
                  ['now_y_%d' % ki for ki in range(num_points)]

        contours = pd.DataFrame(columns=columns)
        contours['barcode'] = position_intissue_onsample['barcode']

        for ni in range(num_points):
            contours['now_x_%d' % (ni)] = np.round(batch_circle_contour[:, ni, 0, 1]).astype(int).tolist()
            contours['now_y_%d' % (ni)] = np.round(batch_circle_contour[:, ni, 0, 0]).astype(int).tolist()

    else:
        # print('aaa')
        single_bbox_contour = np.array([[[ - box_length, - box_length]], [[ - box_length, box_length]],[[box_length, box_length]], [[box_length, - box_length]]])
        batch_bbox_contour = np.zeros((len(point_array_onsample),4, 1, 2))
        batch_bbox_contour[:] = single_bbox_contour
        batch_bbox_contour[:, :, 0, 0] = batch_bbox_contour[:, :, 0, 0] + np.round(((point_array_onsample[:,0] - real_x) / 4)).astype(int)[:, np.newaxis]
        batch_bbox_contour[:, :, 0, 1] = batch_bbox_contour[:, :, 0, 1] + np.round(((point_array_onsample[:,1] - real_y) / 4)).astype(int)[:, np.newaxis]

        contours = pd.DataFrame(columns = ['barcode','now_x_0','now_y_0','now_x_1','now_y_1','now_x_2','now_y_2','now_x_3','now_y_3'])

        contours['barcode'] = position_intissue_onsample['barcode']
        contours['now_x_0'] = np.round(batch_bbox_contour[:, 0, 0, 1]).astype(int).tolist()
        contours['now_x_1'] = np.round(batch_bbox_contour[:, 1, 0, 1]).astype(int).tolist()
        contours['now_x_2'] = np.round(batch_bbox_contour[:, 2, 0, 1]).astype(int).tolist()
        contours['now_x_3'] = np.round(batch_bbox_contour[:, 3, 0, 1]).astype(int).tolist()

        contours['now_y_0'] = np.round(batch_bbox_contour[:, 0, 0, 0]).astype(int).tolist()
        contours['now_y_1'] = np.round(batch_bbox_contour[:, 1, 0, 0]).astype(int).tolist()
        contours['now_y_2'] = np.round(batch_bbox_contour[:, 2, 0, 0]).astype(int).tolist()
        contours['now_y_3'] = np.round(batch_bbox_contour[:, 3, 0, 0]).astype(int).tolist()

    # for ns in range(len(position)):
    # 	now_spot = position.loc[ns]
    # 	in_tissue = int(now_spot['in_tissue'])
    # 	if not in_tissue:
    # 		continue
    # 	# abs_x = int(np.round(now_spot['pxl_row_in_fullres']))
    # 	# abs_y = int(np.round(now_spot['pxl_col_in_fullres']))
    #
    # 	abs_x = int(now_spot['pxl_row_in_fullres'])
    # 	abs_y = int(now_spot['pxl_col_in_fullres'])
    #
    # 	dist = cv2.pointPolygonTest(np.array(ann), (int(abs_y), int(abs_x)), False)
    #
    # 	if dist < 0:
    # 		continue
    #
    # 	now_x = int((abs_x - real_y)/4)
    # 	now_y = int((abs_y - real_x)/4)å
    #
    # 	now_len = len(center)
    # 	center.loc[now_len] = [now_spot['barcode'], abs_x, abs_y, now_x, now_y]

    contours.to_csv(cimg_root.replace('.png', '_%d_%d_spot.csv' % (real_x, real_y)), index = False)


if __name__ == "__main__":
    args, cfg = parse_args(__doc__)
    lv = 0

    Json_folder = cfg['paths']['region_annotation_dir']

    svs_dir = cfg['paths']['wsi_dir']
    output_dir = cfg['paths']['contour_root']

    svs_list = glob.glob(os.path.join(svs_dir, '*.svs'))

    for ki in range(0,len(svs_list)):
        svs = svs_list[ki]
        name = os.path.basename(svs)
        print(name)
        if not '.svs' in name:
            continue
        name_now = name.split('.svs')[0]
        name_list = name.split('.svs')[0].split('-')
        case_name = '%s-%s' % (name_list[0], name_list[1])
        if cfg['cases'] and case_name not in cfg['cases']:
            continue
        IMG = glob.glob(os.path.join(svs_dir,'%s' % (name_now) + '.svs'))[0]
        Json = glob.glob(os.path.join(Json_folder,'*%s' % name_now) + '.svs.geojson')[0]

        now_lv = 0
        num = 0

        draw_line(case_name, IMG, Json, output_dir, now_lv, num, cfg)

