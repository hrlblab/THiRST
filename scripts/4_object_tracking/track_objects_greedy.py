# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Link segmented objects across sections with greedy best-IoU matching (earlier tracker).

THiRST stage 4 - 3D object reconstruction (paper Sec. 4.2.5).

Each track ending on the previous section is continued by the contour of the same class with the
highest IoU (> 0.01).

Every section needs its GeoJSON ``<case>-<section>.svs.geojson`` (an empty FeatureCollection if it has
no object); a missing file stops the script. Objects are not linked across a section without objects:
such sections are named in a warning and the objects after them start new tracks.
Kept because the gene-map and 3D rendering figures were produced from its output.

``--registration patch`` (default) reads ``registration_3D_patch_ANTs/annotation_QA_generegions`` and writes
``Object_IoU_patch/IoU_Object.json``; ``--registration global`` reads ``registration_3D_ANTs/annotation_QA`` and
writes ``Object_IoU_global/IoU_Object.json`` (all under ``<contour_root>/<case>/``).

Usage
-----
    python scripts/4_object_tracking/track_objects_greedy.py --config configs/example.yaml

Original script(s): Step9_contourMatching_object_new.py (patch), Step9_contourMatching_object_global.py (global)
"""
import sys
import os
import glob
import json
import re

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args, section_index, section_object_geojsons  # noqa: E402
from thirst.objects import read_object_contours  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text

# Step 3: Compute IoU between each circle and each bin
def compute_iou(poly1, poly2):
    # Quick bounding box check:
    minx1, miny1, maxx1, maxy1 = poly1.bounds
    minx2, miny2, maxx2, maxy2 = poly2.bounds
    if (maxx1 < minx2) or (maxx2 < minx1) or (maxy1 < miny2) or (maxy2 < miny1):
        return 0.0  # No overlap in bounding boxes

    # Now compute the detailed intersection/union
    inter = poly1.intersection(poly2)
    if inter.is_empty:
        return 0.0
    inter_area = inter.area
    union_area = poly1.area + poly2.area - inter_area
    return inter_area / union_area if union_area else 0.0


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def load_annotations(image_name, mask_json_path, mask_labels):
    """
    Loads a GeoJSON file, extracts contours for given object labels,
    applies a translation transformation based on the image filename,
    and returns a dictionary mapping object type to a list of numpy arrays (contours).
    """
    if not os.path.exists(mask_json_path):
        print(f"Skipping {mask_json_path}, file not found")
        return None

    # {class: [contour, ...]}; reads standard GeoJSON and the earlier per-class layout
    return read_object_contours(mask_json_path, mask_labels)


def contours_to_polygons(annotations_by_object):
    """
    Converts each contour (numpy array) into a Shapely Polygon.
    Returns a list of dictionaries for the slice,
    each with keys: 'object' (type), 'index' (contour index) and 'polygon' (Shapely Polygon).
    """
    slice_contours = []
    for obj, contour_list in annotations_by_object.items():
        for idx, pts in enumerate(contour_list):
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    # Optionally, fix invalid polygons (e.g., self-intersections)
                    poly = poly.buffer(0)
                slice_contours.append({'object': obj, 'index': idx, 'polygon': poly})
            except Exception as e:
                print("Error converting contour to polygon:", e)
    return slice_contours


from shapely.strtree import STRtree


def save_tracks_to_json(tracks, output_file):
    """
    Converts tracks into a JSON-serializable structure and writes them to a file.
    Each track is a dictionary containing the track id and a list of contour entries.
    Each contour entry includes the slice index, object type, contour index, and the polygon coordinates.
    """
    output_data = []
    for track_id, track in tracks.items():
        track_entry = {
                'track_id': track_id,
                'contours': []
        }
        for entry in track:
            # Convert the polygon to a list of (x,y) tuples.
            # You can also use WKT via entry['polygon'].wkt if preferred.
            coords = list(entry['polygon'].exterior.coords) if entry['polygon'].exterior is not None else []
            track_entry['contours'].append({
                    'slice_index': entry.get('slice_index'),
                    'object': entry.get('object'),
                    'index': entry.get('index'),
                    'coordinates': coords
            })
        output_data.append(track_entry)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Tracks saved to {output_file}")


def match_tracks(previous_tracks, current_contours, iou_threshold=0.5):
    # Build an STRtree for current contours for fast spatial queries.
    current_polygons = [d['polygon'] for d in current_contours]
    tree = STRtree(current_polygons)

    assigned_indices = set()

    for track_id, track_data in previous_tracks.items():
        last_contour = track_data[-1]
        poly = last_contour['polygon']
        # The query now returns indices.
        candidate_indices = tree.query(poly, predicate='intersects')

        best_candidate = None
        best_iou = 0.0

        for candidate_index in candidate_indices:
            candidate_index = int(candidate_index)  # Ensure it's a standard int.
            candidate_dict = current_contours[candidate_index]
            candidate_poly = candidate_dict['polygon']
            if candidate_index in assigned_indices or candidate_dict['object'] != last_contour['object']:
                continue   # already taken, or another object class

            iou = compute_iou(poly, candidate_poly)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_candidate = candidate_index

        if best_candidate is not None:
            track_data.append(current_contours[best_candidate])
            assigned_indices.add(best_candidate)

    unmatched = [contour for ci, contour in enumerate(current_contours) if ci not in assigned_indices]
    return previous_tracks, unmatched

if __name__ == '__main__':
    args, cfg = parse_args(__doc__, lambda p: p.add_argument(
        '--registration', choices=('patch', 'global'), default='patch',
        help="track objects after patch-level refinement (default) or after global registration only"))
    case_folder =  cfg['paths']['image_root']
    point_folder = cfg['paths']['point_root']
    contour_folder = cfg['paths']['contour_root']

    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        # image_badcase_dir = os.path.join(case, 'registration_badcase_2048')
        reconstruction_dir = os.path.join(case.replace(case_folder,point_folder), 'registration_3D_patch_ANTs')
        reconstruction_contour_dir = os.path.join(case.replace(case_folder,contour_folder), 'registration_3D_patch_ANTs')
        if args.registration == 'patch':
            reconstruction_object_dir = os.path.join(case.replace(case_folder,contour_folder),'registration_3D_patch_ANTs', 'annotation_QA_generegions')
            IoU_dir = os.path.join(case.replace(case_folder,contour_folder), 'Object_IoU_patch')
        else:  # global registration only
            reconstruction_object_dir = os.path.join(case.replace(case_folder,contour_folder), 'registration_3D_ANTs', 'annotation_QA')
            IoU_dir = os.path.join(case.replace(case_folder,contour_folder), 'Object_IoU_global')


        if not os.path.exists(IoU_dir):
            os.makedirs(IoU_dir)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)

        jsons = section_object_geojsons(images, reconstruction_object_dir, now_case)  # one per section, in stack order

        middle_idx = int(len(images) / 2)

        # print(middle_idx)
        # middle_image = plt.imread(images[middle_idx])[:,:,:3]

        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        patch_level_folder = os.path.join(case,'3D_Patch_crop')

        patch_list = glob.glob(os.path.join(patch_level_folder,'*'))

        patch_array = np.zeros((len(patch_list), 1, 2)).astype(np.float32)
        for pi in range(len(patch_list)):
            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi,:,0] = int(patch_abs_x)
            patch_array[pi,:,1] = int(patch_abs_y)

        mask_labels = list(cfg['object_classes'])  # Omni-Seg classes reconstructed in 3D
        # Build a dictionary that holds all contours (as polygons) for each slice
        all_slices = {}  # key: slice index, value: list of dicts with keys 'object' and 'polygon'
        for idx, (img_name, json_path) in enumerate(zip(images, jsons)):
            ann = load_annotations(img_name, json_path, mask_labels)
            all_slices[idx] = contours_to_polygons(ann)
        empty = [section_index(images[i]) for i in range(len(images)) if not all_slices[i]]
        if empty:
            print(f"[WARN] {now_case}: no {mask_labels} objects on section(s) {empty}; "
                  f"objects cannot be linked across these sections (tracks restart after them)")

        tracks = {}  # e.g., { track_id: [(slice_index, contour), ...], ... }
        next_track_id = 0
        IOU_THRESHOLD = 0.01


        first_slice = all_slices[0]
        for contour in first_slice:
            contour['slice_index'] = 0
            tracks[next_track_id] = [contour]
            next_track_id += 1

        # Process each subsequent slice.
        for slice_index in range(1, len(images)):   # every section, including empty ones
            previous_num = next_track_id
            print('now slice: ', slice_index)
            print('previous tracks num: ', previous_num)
            current_contours = all_slices[slice_index]
            for contour in current_contours:
                contour['slice_index'] = slice_index

            # Filter out tracks whose last contour comes from the previous slice.
            previous_tracks = {tid: track for tid, track in tracks.items()
                                               if track[-1]['slice_index'] == slice_index - 1}

            # Update the matching tracks and get unmatched contours.
            updated_tracks, unmatched_contours = match_tracks(previous_tracks, current_contours, iou_threshold=0.01)

            # (Since updated_tracks are references to the entries in 'tracks', 'tracks' is already updated.)
            # Start new tracks for the unmatched contours.
            for contour in unmatched_contours:
                tracks[next_track_id] = [contour]
                next_track_id += 1
            print('now objects ', len(current_contours))
            print('previous now num: ', len(tracks))
            now_num = next_track_id
            print('tracks add ', now_num - previous_num)

        # # # Example: print out the final tracks.
        # print("Final tracks:")
        # for track_id, track in tracks.items():
        # 	print(f"Track {track_id}:")
        # 	for entry in track:
        # 		print(f"   Slice {entry['slice_index']} - Object {entry['object']} - Index {entry['index']}")
        output_file = os.path.join(IoU_dir, 'IoU_Object.json')
        save_tracks_to_json(tracks, output_file)