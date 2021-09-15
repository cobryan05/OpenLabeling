#!/bin/python
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import czifile
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

sys.path.append( os.path.join( os.path.dirname( __file__ ), "..", "submodules" ) )

from trackerTools.objectTracker import ObjectTracker
from trackerTools.utils import *

from lxml import etree
import xml.etree.cElementTree as ET

DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Open-source image labeling tool')
parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
parser.add_argument('-t', '--thickness', default='1', type=int, help='Bounding box and cross line thickness')
parser.add_argument('-y', '--yoloWeights', default='', type=str, help='YOLO Weights file to use for prediction')
parser.add_argument('--draw-from-PASCAL-files', action='store_true', help='Draw bounding boxes from the PASCAL files') # default YOLO
'''
tracker_types = ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']
    Recomended tracker_type:
        DASIAMRPN -> best
        KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
        CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
        MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
'''
parser.add_argument('--tracker', default='KCF', type=str, help="tracker_type being used: ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']")
parser.add_argument('-n', '--n_frames', default='200', type=int, help='number of frames to track object for')
args = parser.parse_args()

gClassIdx = 0
gImgIdx = 0
gOrigImg = None
gImgObjects = []

INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir
N_FRAMES   = args.n_frames
TRACKER_TYPE = args.tracker

if TRACKER_TYPE == "DASIAMRPN":
    from dasiamrpn import dasiamrpn

WINDOW_NAME    = 'OpenLabeling'
ZOOM_WINDOW_NAME = 'OpenLabeling (Zoomed)'
TRACKBAR_IMG   = 'Image'
TRACKBAR_CLASS = 'Class'

annotation_formats = {'YOLO_darknet' : '.txt'}
TRACKER_DIR = os.path.join(OUTPUT_DIR, '.tracker')

DRAW_FROM_PASCAL = args.draw_from_PASCAL_files

# selected bounding box
gPrevWasDoubleClick = False
gIsBboxSelected = False
gSelectedBbox = -1
LINE_THICKNESS = args.thickness
ZOOM_RADIUS = 100
ZOOM_STEP = 5
ZOOM_MIN = 50

gMouseX = 0
gMouseY = 0
gPoint1 = (-1, -1)
gPoint2 = (-1, -1)

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


# Check if a point belongs to a rectangle
def pointInRect(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
    return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom



# Class to deal with bbox resizing
class dragBBox:
    '''
        LT -- MT -- RT
        |            |
        LM          RM
        |            |
        LB -- MB -- RB
    '''

    # Size of resizing anchors (depends on LINE_THICKNESS)
    sRA = LINE_THICKNESS * 2

    # Object being dragged
    selected_object = None

    # Flag indicating which resizing-anchor is dragged
    anchor_being_dragged = None

    '''
    \brief This method is used to check if a current mouse position is inside one of the resizing anchors of a bbox
    '''
    @staticmethod
    def check_point_inside_resizing_anchors(eX, eY, obj):
        _class_name, x_left, y_top, x_right, y_bottom = obj
        # first check if inside the bbox region (to avoid making 8 comparisons per object)
        if pointInRect(eX, eY,
                        x_left - dragBBox.sRA,
                        y_top - dragBBox.sRA,
                        x_right + dragBBox.sRA,
                        y_bottom + dragBBox.sRA):

            anchor_dict = get_anchors_rectangles(x_left, y_top, x_right, y_bottom)
            for anchor_key in anchor_dict:
                rX_left, rY_top, rX_right, rY_bottom = anchor_dict[anchor_key]
                if pointInRect(eX, eY, rX_left, rY_top, rX_right, rY_bottom):
                    dragBBox.anchor_being_dragged = anchor_key
                    break

    '''
    \brief This method is used to select an object if one presses a resizing anchor
    '''
    @staticmethod
    def handler_left_mouse_down(eX, eY, obj):
        dragBBox.check_point_inside_resizing_anchors(eX, eY, obj)
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.selected_object = obj

    @staticmethod
    def handler_mouse_move(eX, eY):
        global gRedrawNeeded
        if dragBBox.selected_object is not None:
            class_name, x_left, y_top, x_right, y_bottom = dragBBox.selected_object

            # Do not allow the bbox to flip upside down (given a margin)
            margin = 3 * dragBBox.sRA
            change_was_made = False

            if dragBBox.anchor_being_dragged[0] == "L":
                # left anchors (LT, LM, LB)
                if eX < x_right - margin:
                    x_left = eX
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[0] == "R":
                # right anchors (RT, RM, RB)
                if eX > x_left + margin:
                    x_right = eX
                    change_was_made = True

            if dragBBox.anchor_being_dragged[1] == "T":
                # top anchors (LT, RT, MT)
                if eY < y_bottom - margin:
                    y_top = eY
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[1] == "B":
                # bottom anchors (LB, RB, MB)
                if eY > y_top + margin:
                    y_bottom = eY
                    change_was_made = True

            if change_was_made:
                action = "resize_bbox:{}:{}:{}:{}".format(x_left, y_top, x_right, y_bottom)
                edit_bbox(dragBBox.selected_object, action)
                gRedrawNeeded = True
                # update the selected bbox
                dragBBox.selected_object = [class_name, x_left, y_top, x_right, y_bottom]

    '''
    \brief This method will reset this class
     '''
    @staticmethod
    def handler_left_mouse_up(eX, eY):
        if dragBBox.selected_object is not None:
            dragBBox.selected_object = None
            dragBBox.anchor_being_dragged = None

def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)

def get_img_path( idx = None ):
    global gImgIdx
    idx = idx or gImgIdx
    return IMAGE_PATH_LIST[idx]

def set_img_index(x):
    global gImgIdx, gOrigImg, gImgObjects, gIsBboxSelected, gRedrawNeeded
    gImgIdx = x
    img_path = get_img_path()
    gOrigImg = cv2.imread(img_path)
    gIsBboxSelected = False
    gRedrawNeeded = True
    text = 'Showing image {}/{}, path: {}'.format(str(gImgIdx), str(last_img_index), img_path)
    display_text(text, 1000)

    # create empty annotation files for each image, if it doesn't exist already
    abs_path = os.path.abspath(img_path)
    folder_name = os.path.dirname(img_path)
    image_name = os.path.basename(img_path)
    img_height, img_width, depth = (str(number) for number in gOrigImg.shape)

    annotation_paths = get_annotation_paths(img_path, annotation_formats)
    gImgObjects = read_objects_from_file( annotation_paths )
    for ann_path in annotation_paths:
        if not os.path.isfile(ann_path):
            if '.txt' in ann_path:
                open(ann_path, 'a').close()
            elif '.xml' in ann_path:
                create_PASCAL_VOC_xml(ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)

def get_vid_img_index( curIdx, dir, lastIdx ):
    startIdx = curIdx
    if dir > 0:
        op = increase_index
    elif dir < 0:
        op = decrease_index
    else:
        return startIdx

    prev_video_name = None
    while True:
        img_path = get_img_path(curIdx)
        is_from_video, video_name = is_frame_from_video(img_path)

        # If video name changed, then break
        if prev_video_name is not None and video_name != prev_video_name:
            break

        curIdx = op( curIdx, lastIdx )

        # If looped around or not a video, then break
        if curIdx == startIdx or not is_from_video:
            break
        prev_video_name = video_name
    return curIdx


def set_class_index(x):
    global gClassIdx
    gClassIdx = x
    text = 'Selected class {}/{} -> {}'.format(str(gClassIdx), str(last_class_index), CLASS_LIST[gClassIdx])
    display_text(text, 3000)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 15, 25, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)


def yolo_format(class_index, point_1, point_2, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = float((point_1[0] + point_2[0]) / (2.0 * width) )
    y_center = float((point_1[1] + point_2[1]) / (2.0 * height))
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    items = map(str, [class_index, x_center, y_center, x_width, y_height])
    return ' '.join(items)


def voc_format(class_name, point_1, point_2):
    # Order: class_name xmin ymin xmax ymax
    xmin, ymin = min(point_1[0], point_2[0]), min(point_1[1], point_2[1])
    xmax, ymax = max(point_1[0], point_2[0]), max(point_1[1], point_2[1])
    items = map(str, [class_name, xmin, ymin, xmax, ymax])
    return items

def findIndex(obj_to_find):
    #return [(ind, img_objects[ind].index(obj_to_find)) for ind in xrange(len(img_objects)) if item in img_objects[ind]]
    ind = -1
    obj_to_find = list(obj_to_find) # Handle np arrays being passed in
    ind_ = 0
    for listElem in gImgObjects:
        if listElem == obj_to_find:
            ind = ind_
            return ind
        ind_ = ind_+1

    return ind

def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


def append_bb(ann_path, line, extension):
    global gRedrawNeeded
    gRedrawNeeded = True

    if '.txt' in extension:
        with open(ann_path, 'a') as myfile:
            myfile.write(line + '\n') # append line
    elif '.xml' in extension:
        class_name, xmin, ymin, xmax, ymax = line

        tree = ET.parse(ann_path)
        annotation = tree.getroot()

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = xmin
        ET.SubElement(bbox, 'ymin').text = ymin
        ET.SubElement(bbox, 'xmax').text = xmax
        ET.SubElement(bbox, 'ymax').text = ymax

        xml_str = ET.tostring(annotation)
        write_xml(xml_str, ann_path)


def yolo_to_voc(x_center, y_center, x_width, y_height, width, height):
    x_center *= float(width)
    y_center *= float(height)
    x_width *= float(width)
    y_height *= float(height)
    x_width /= 2.0
    y_height /= 2.0
    xmin = int(round(x_center - x_width))
    ymin = int(round(y_center - y_height))
    xmax = int(round(x_center + x_width))
    ymax = int(round(y_center + y_height))
    return xmin, ymin, xmax, ymax


def get_xml_object_data(obj):
    class_name = obj.find('name').text
    class_index = CLASS_LIST.index(class_name)
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    return [class_name, class_index, xmin, ymin, xmax, ymax]


def get_txt_object_data(obj, img_width, img_height):
    classId, centerX, centerY, bbox_width, bbox_height = obj.split()
    bbox_width = float(bbox_width)
    bbox_height  = float(bbox_height)
    centerX = float(centerX)
    centerY = float(centerY)

    class_index = int(classId)
    class_name = CLASS_LIST[class_index]
    xmin = int(img_width * centerX - img_width * bbox_width/2.0)
    xmax = int(img_width * centerX + img_width * bbox_width/2.0)
    ymin = int(img_height * centerY - img_height * bbox_height/2.0)
    ymax = int(img_height * centerY + img_height * bbox_height/2.0)
    return [class_name, class_index, xmin, ymin, xmax, ymax]


def get_anchors_rectangles(xmin, ymin, xmax, ymax):
    anchor_list = {}

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    L_ = [xmin - dragBBox.sRA, xmin + dragBBox.sRA]
    M_ = [mid_x - dragBBox.sRA, mid_x + dragBBox.sRA]
    R_ = [xmax - dragBBox.sRA, xmax + dragBBox.sRA]
    _T = [ymin - dragBBox.sRA, ymin + dragBBox.sRA]
    _M = [mid_y - dragBBox.sRA, mid_y + dragBBox.sRA]
    _B = [ymax - dragBBox.sRA, ymax + dragBBox.sRA]

    anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
    anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
    anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
    anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
    anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
    anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
    anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
    anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

    return anchor_list


def draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color):
    anchor_dict = get_anchors_rectangles(xmin, ymin, xmax, ymax)
    for anchor_key in anchor_dict:
        x1, y1, x2, y2 = anchor_dict[anchor_key]
        cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return tmp_img

def draw_bboxes_from_file(tmp_img, annotation_paths, class_filter = None ):
    global gImgObjects
    gImgObjects = read_objects_from_file( annotation_paths )
    return draw_bboxes( tmp_img, gImgObjects, class_filter = class_filter)

def read_objects_from_file( annotation_paths ):
    global gOrigImg
    fileObjects = []
    ann_path = None
    if DRAW_FROM_PASCAL:
        # Read objects from PASCAL file
        ann_path = next(path for path in annotation_paths if 'PASCAL_VOC' in path)
    else:
        # Read objects from YOLO file
        ann_path = next(path for path in annotation_paths if 'YOLO_darknet' in path)

    if os.path.isfile(ann_path):
        if DRAW_FROM_PASCAL:
            tree = ET.parse(ann_path)
            annotation = tree.getroot()
            for idx, obj in enumerate(annotation.findall('object')):
                class_name, class_index, xmin, ymin, xmax, ymax = get_xml_object_data(obj)
                #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                fileObjects.append([class_index, xmin, ymin, xmax, ymax])
        else:
            imgHeight, imgWidth = gOrigImg.shape[:2]
            # Draw from YOLO
            with open(ann_path) as fp:
                for idx, line in enumerate(fp):
                    obj = line
                    class_name, class_index, xmin, ymin, xmax, ymax = get_txt_object_data(obj, imgWidth, imgHeight)
                    #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                    fileObjects.append([class_index, xmin, ymin, xmax, ymax])
    return fileObjects


def draw_bboxes( img, objects, class_filter = None ):
    for idx,obj in enumerate(objects):
        class_idx, x1, y1, x2, y2 = obj
        color = class_rgb[class_idx].tolist()

        # draw resizing anchors if the object is selected
        if gIsBboxSelected:
            if idx == gSelectedBbox:
                img = draw_bbox_anchors(img, x1, y1, x2, y2, color)

        # draw bbox
        if class_filter is None or class_filter == class_idx:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_THICKNESS)
            if text_on:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, CLASS_LIST[class_idx], (x1, y1 - 5), font, 0.6, color, LINE_THICKNESS, cv2.LINE_AA)
    return img



def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height


def set_selected_bbox(set_class):
    global gIsBboxSelected, gSelectedBbox, gRedrawNeeded
    smallest_area = -1
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(gImgObjects):
        ind, x1, y1, x2, y2 = obj
        x1 = x1 - dragBBox.sRA
        y1 = y1 - dragBBox.sRA
        x2 = x2 + dragBBox.sRA
        y2 = y2 + dragBBox.sRA
        if pointInRect(gMouseX, gMouseY, x1, y1, x2, y2):
            gIsBboxSelected = True
            gRedrawNeeded = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                gSelectedBbox = idx
                if set_class:
                    # set class to the one of the selected bounding box
                    cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, ind)


def is_mouse_inside_delete_button():
    for idx, obj in enumerate(gImgObjects):
        if idx == gSelectedBbox:
            _ind, x1, y1, x2, y2 = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if pointInRect(gMouseX, gMouseY, x1_c, y1_c, x2_c, y2_c):
                return True
    return False


def edit_bbox(obj_to_edit, action):
    ''' action = `delete`
                 `change_class:new_class_index`
                 `resize_bbox:new_x_left:new_y_top:new_x_right:new_y_bottom`
    '''
    if 'change_class' in action:
        new_class_index = int(action.split(':')[1])
    elif 'resize_bbox' in action:
        new_x_left = max(0, int(action.split(':')[1]))
        new_y_top = max(0, int(action.split(':')[2]))
        new_x_right = min(width, int(action.split(':')[3]))
        new_y_bottom = min(height, int(action.split(':')[4]))

    # 1. initialize bboxes_to_edit_dict
    #    (we use a dict since a single label can be associated with multiple ones in videos)
    bboxes_to_edit_dict = {}
    current_img_path = get_img_path()
    bboxes_to_edit_dict[current_img_path] = obj_to_edit

    # 2. add elements to bboxes_to_edit_dict
    '''
        If the bbox is in the json file then it was used by the video Tracker, hence,
        we must also edit the next predicted bboxes associated to the same `anchor_id`.
    '''
    # if `current_img_path` is a frame from a video
    is_from_video, video_name = is_frame_from_video(current_img_path)
    if is_from_video:
        # get json file corresponding to that video
        json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
        file_exists, json_file_data = get_json_file_data(json_file_path)
        # if json file exists
        if file_exists:
            # match obj_to_edit with the corresponding json object
            frame_data_dict = json_file_data['frame_data_dict']
            json_object_list = get_json_file_object_list(current_img_path, frame_data_dict)
            obj_matched = get_json_object_dict(obj_to_edit, json_object_list)
            # if match found
            if obj_matched is not None:
                # get this object's anchor_id
                anchor_id = obj_matched['anchor_id']

                frame_path_list = get_next_frame_path_list(video_name, current_img_path)
                frame_path_list.insert(0, current_img_path)

                if 'change_class' in action:
                    # add also the previous frames
                    prev_path_list = get_prev_frame_path_list(video_name, current_img_path)
                    frame_path_list = prev_path_list + frame_path_list

                # update json file if contain the same anchor_id
                for frame_path in frame_path_list:
                    json_object_list = get_json_file_object_list(frame_path, frame_data_dict)
                    json_obj = get_json_file_object_by_id(json_object_list, anchor_id)
                    if json_obj is not None:
                        bboxes_to_edit_dict[frame_path] = [
                            json_obj['class_index'],
                            json_obj['bbox']['xmin'],
                            json_obj['bbox']['ymin'],
                            json_obj['bbox']['xmax'],
                            json_obj['bbox']['ymax']
                        ]
                        # edit json file
                        if 'delete' in action:
                            json_object_list.remove(json_obj)
                        elif 'change_class' in action:
                            json_obj['class_index'] = new_class_index
                        elif 'resize_bbox' in action:
                            json_obj['bbox']['xmin'] = new_x_left
                            json_obj['bbox']['ymin'] = new_y_top
                            json_obj['bbox']['xmax'] = new_x_right
                            json_obj['bbox']['ymax'] = new_y_bottom
                    else:
                        break

                # save the edited data
                with open(json_file_path, 'w') as outfile:
                    json.dump(json_file_data, outfile, sort_keys=True, indent=4)

    # 3. loop through bboxes_to_edit_dict and edit the corresponding annotation files
    for path in bboxes_to_edit_dict:
        obj_to_edit = bboxes_to_edit_dict[path]
        class_index, xmin, ymin, xmax, ymax = map(int, obj_to_edit)

        for ann_path in get_annotation_paths(path, annotation_formats):
            if '.txt' in ann_path:
                # edit YOLO file
                with open(ann_path, 'r') as old_file:
                    lines = old_file.readlines()

                yolo_line = yolo_format(class_index, (xmin, ymin), (xmax, ymax), width, height) # TODO: height and width ought to be stored
                ind = findIndex(obj_to_edit)
                i=0

                with open(ann_path, 'w') as new_file:
                    for line in lines:

                        if i != ind:
                           new_file.write(line)

                        elif 'change_class' in action:
                            new_yolo_line = yolo_format(new_class_index, (xmin, ymin), (xmax, ymax), width, height)
                            new_file.write(new_yolo_line + '\n')
                        elif 'resize_bbox' in action:
                            new_yolo_line = yolo_format(class_index, (new_x_left, new_y_top), (new_x_right, new_y_bottom), width, height)
                            new_file.write(new_yolo_line + '\n')

                        i=i+1

            elif '.xml' in ann_path:
                # edit PASCAL VOC file
                tree = ET.parse(ann_path)
                annotation = tree.getroot()
                for obj in annotation.findall('object'):
                    class_name_xml, class_index_xml, xmin_xml, ymin_xml, xmax_xml, ymax_xml = get_xml_object_data(obj)
                    if ( class_index == class_index_xml and
                                     xmin == xmin_xml and
                                     ymin == ymin_xml and
                                     xmax == xmax_xml and
                                     ymax == ymax_xml ) :
                        if 'delete' in action:
                            annotation.remove(obj)
                        elif 'change_class' in action:
                            # edit object class name
                            object_class = obj.find('name')
                            object_class.text = CLASS_LIST[new_class_index]
                        elif 'resize_bbox' in action:
                            object_bbox = obj.find('bndbox')
                            object_bbox.find('xmin').text = str(new_x_left)
                            object_bbox.find('ymin').text = str(new_y_top)
                            object_bbox.find('xmax').text = str(new_x_right)
                            object_bbox.find('ymax').text = str(new_y_bottom)
                        break

                xml_str = ET.tostring(annotation)
                write_xml(xml_str, ann_path)


def mouse_listener(event, x, y, flags, param):
    # mouse callback function
    global gIsBboxSelected, gPrevWasDoubleClick, gMouseX, gMouseY, gPoint1, gPoint2, gRedrawNeeded, gOrigImg

    if event == cv2.EVENT_MOUSEMOVE:
        gMouseX = x
        gMouseY = y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        gPrevWasDoubleClick = True
        #print('Double click')
        gPoint1 = (-1, -1)
        # if clicked inside a bounding box we set that bbox
        set_selected_bbox(True)
    # By AlexeyGy: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_selected_bbox(False)
        if gIsBboxSelected:
            obj_to_edit = gImgObjects[gSelectedBbox]
            edit_bbox(obj_to_edit, 'delete')
            gIsBboxSelected = False
            gRedrawNeeded = True
    # Change class to current class via middle-click
    elif event == cv2.EVENT_MBUTTONDOWN:
        set_selected_bbox(False)
        if gIsBboxSelected:
            obj_to_edit = gImgObjects[gSelectedBbox]
            edit_bbox(obj_to_edit, f'change_class:{gClassIdx}')
            gIsBboxSelected = False
            gRedrawNeeded = True

    elif event == cv2.EVENT_LBUTTONDOWN:
        if gPrevWasDoubleClick:
            #print('Finish double click')
            gPrevWasDoubleClick = False
        else:
            #print('Normal left click')

            # Check if mouse inside on of resizing anchors of the selected bbox
            if gIsBboxSelected:
                dragBBox.handler_left_mouse_down(x, y, gImgObjects[gSelectedBbox])

            if dragBBox.anchor_being_dragged is None:
                if gPoint1[0] == -1:
                    if gIsBboxSelected:
                        if is_mouse_inside_delete_button():
                            set_selected_bbox(True)
                            obj_to_edit = gImgObjects[gSelectedBbox]
                            edit_bbox(obj_to_edit, 'delete')
                        gIsBboxSelected = False
                    else:
                        # first click (start drawing a bounding box or delete an item)

                        gPoint1 = (x, y)
                else:
                    # minimal size for bounding box to avoid errors
                    threshold = 5
                    if abs(x - gPoint1[0]) > threshold or abs(y - gPoint1[1]) > threshold:
                        # second click
                        gPoint2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_left_mouse_up(x, y)



def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img


def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(gImgObjects):
        ind, x1, y1, x2, y2 = obj
        # if idx == selected_bbox:
        #     x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
        #     draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def convert_video_to_images(video_path, n_frames, desired_img_format):
    # create folder to store images (if video was not converted to images already)
    file_path, file_extension = os.path.splitext(video_path)
    # append extension to avoid collision of videos with same name
    # e.g.: `video.mp4`, `video.avi` -> `video_mp4/`, `video_avi/`
    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    video_name_ext = os.path.basename(file_path)
    if not os.path.exists(file_path):
        print(' Converting video to individual frames...')
        cap = cv2.VideoCapture(video_path)
        os.makedirs(file_path)
        # read the video
        for i in tqdm(range(n_frames)):
            if not cap.isOpened():
                break
            # capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # save each frame (we use this format to avoid repetitions)
                frame_name =  '{}_{}{}'.format(video_name_ext, i, desired_img_format)
                frame_path = os.path.join(file_path, frame_name)
                cv2.imwrite(frame_path, frame)
        # release the video capture object
        cap.release()
    return file_path, video_name_ext


def convert_czi_to_image(czi_path, desired_img_format):
    # create folder to store images

    split_path = czi_path.split(os.path.sep)
    input_folder = split_path[0]
    output_folder = os.path.join( input_folder, "converted" )

    converted_name, converted_ext = os.path.splitext( czi_path )
    converted_ext = converted_ext.replace(".", "_")
    converted_name = converted_name.replace("\\", "_") + converted_ext + desired_img_format
    converted_path = os.path.join( output_folder, converted_name )

    if not os.path.exists(converted_path):
        os.makedirs(output_folder, exist_ok=True)

        print(f"Converting czi image {czi_path} -> {converted_name}")
        img = czifile.imread( czi_path )
        img = img.reshape( img.shape[1:] )
        ret = cv2.imwrite( converted_path, img )
    return converted_path

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def get_annotation_paths(img_path, annotation_formats):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(OUTPUT_DIR, ann_dir)
        new_path = os.path.join(new_path, os.path.basename(os.path.normpath(img_path))) #img_path.replace(INPUT_DIR, new_path, 1)
        pre_path, img_ext = os.path.splitext(new_path)
        new_path = new_path.replace(img_ext, ann_ext, 1)
        annotation_paths.append(new_path)
    return annotation_paths


def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth):
    # By: Jatin Kumar Mandav
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = abs_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = img_width
    ET.SubElement(size, 'height').text = img_height
    ET.SubElement(size, 'depth').text = depth
    ET.SubElement(annotation, 'segmented').text = '0'

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, xml_path)


def save_bounding_box(annotation_paths, class_index, point_1, point_2, imgWidth, imgHeight):
    for ann_path in annotation_paths:
        if '.txt' in ann_path:
            line = yolo_format(class_index, point_1, point_2, imgWidth, imgHeight)
            append_bb(ann_path, line, '.txt')
        elif '.xml' in ann_path:
            line = voc_format(CLASS_LIST[class_index], point_1, point_2)
            append_bb(ann_path, line, '.xml')

def is_frame_from_video(img_path):
    img_path_parts = os.path.normpath(img_path).split(os.sep)
    for video_name in VIDEO_NAME_DICT:
        if video_name in img_path_parts:
            # image belongs to a video
            return True, video_name
    return False, None


def get_json_file_data(json_file_path):
    if os.path.isfile(json_file_path):
        with open(json_file_path) as f:
            data = json.load(f)
            return True, data
    else:
        return False, {'n_anchor_ids':0, 'frame_data_dict':{}}


def get_prev_frame_path_list(video_name, img_path):
    first_index = VIDEO_NAME_DICT[video_name]['first_index']
    last_index = VIDEO_NAME_DICT[video_name]['last_index']
    img_index = IMAGE_PATH_LIST.index(img_path)
    return IMAGE_PATH_LIST[first_index:img_index]


def get_next_frame_path_list(video_name, img_path):
    first_index = VIDEO_NAME_DICT[video_name]['first_index']
    last_index = VIDEO_NAME_DICT[video_name]['last_index']
    img_index = IMAGE_PATH_LIST.index(img_path)
    return IMAGE_PATH_LIST[(img_index + 1):last_index]


def get_json_object_dict(obj, json_object_list):
    if len(json_object_list) > 0:
        class_index, xmin, ymin, xmax, ymax = map(int, obj)
        for d in json_object_list:
                    if ( d['class_index'] == class_index and
                         d['bbox']['xmin'] == xmin and
                         d['bbox']['ymin'] == ymin and
                         d['bbox']['xmax'] == xmax and
                         d['bbox']['ymax'] == ymax ) :
                        return d
    return None


def remove_already_tracked_objects(object_list, img_path, json_file_data):
    frame_data_dict = json_file_data['frame_data_dict']
    json_object_list = get_json_file_object_list(img_path, frame_data_dict)
    # copy the list since we will be deleting elements without restarting the loop
    temp_object_list = object_list[:]
    for obj in temp_object_list:
        obj_dict = get_json_object_dict(obj, json_object_list)
        if obj_dict is not None:
            object_list.remove(obj)
            json_object_list.remove(obj_dict)
    return object_list


def get_json_file_object_by_id(json_object_list, anchor_id):
    for obj_dict in json_object_list:
        if obj_dict['anchor_id'] == anchor_id:
            return obj_dict
    return None


def get_json_file_object_list(img_path, frame_data_dict):
    object_list = []
    if img_path in frame_data_dict:
        object_list = frame_data_dict[img_path]
    return object_list


def json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj):
    object_list = get_json_file_object_list(img_path, frame_data_dict)
    class_index, xmin, ymin, xmax, ymax = obj

    bbox = {
      'xmin': xmin,
      'ymin': ymin,
      'xmax': xmax,
      'ymax': ymax
    }

    temp_obj = {
      'anchor_id': anchor_id,
      'prediction_index': pred_counter,
      'class_index': class_index,
      'bbox': bbox
    }

    object_list.append(temp_obj)
    frame_data_dict[img_path] = object_list

    return frame_data_dict


class LabelTracker():
    ''' Special thanks to Rafael Caballero Gonzalez '''
    # extract the OpenCV version info, e.g.:
    # OpenCV 3.3.4 -> [major_ver].[minor_ver].[subminor_ver]
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # TODO: press ESC to stop the tracking process

    def __init__(self, tracker_type, init_frame, next_frame_path_list):
        tracker_types = ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']
        ''' Recomended tracker_type:
              KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
              CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
              MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
        '''
        self.tracker_type = tracker_type
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if tracker_type == tracker_types[0] or tracker_type == tracker_types[2]:
            if int(self.major_ver == 3) and int(self.minor_ver) < 4:
                self.tracker_type = tracker_types[1] # Use KCF instead of CSRT or MOSSE
        # --
        self.init_frame = init_frame
        self.next_frame_path_list = next_frame_path_list

        self.img_h, self.img_w = init_frame.shape[:2]


    def call_tracker_constructor(self, tracker_type):
        if tracker_type == 'DASIAMRPN':
            tracker = dasiamrpn()
        else:
            # -- TODO: remove this if I assume OpenCV version > 3.4.0
            if int(self.major_ver == 3) and int(self.minor_ver) < 3:
                #tracker = cv2.Tracker_create(tracker_type)
                pass
            # --
            else:
                try:
                    tracker = cv2.TrackerKCF_create()
                except AttributeError as error:
                    print(error)
                    print('\nMake sure that OpenCV contribute is installed: opencv-contrib-python\n')
                if tracker_type == 'CSRT':
                    tracker = cv2.TrackerCSRT_create()
                elif tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                elif tracker_type == 'MOSSE':
                    tracker = cv2.TrackerMOSSE_create()
                elif tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                elif tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                elif tracker_type == 'MEDIANFLOW':
                    tracker = cv2.TrackerMedianFlow_create()
                elif tracker_type == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                elif tracker_type == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()
        return tracker


    def start_tracker(self, json_file_data, json_file_path, img_path, obj, color, annotation_formats):
        tracker = self.call_tracker_constructor(self.tracker_type)
        anchor_id = json_file_data['n_anchor_ids']
        frame_data_dict = json_file_data['frame_data_dict']

        pred_counter = 0
        frame_data_dict = json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj)
        # tracker bbox format: xmin, xmax, w, h
        xmin, ymin, xmax, ymax = obj[1:5]
        initial_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        tracker.init(self.init_frame, initial_bbox)
        for frame_path in self.next_frame_path_list:
            next_image = cv2.imread(frame_path)
            # get the new bbox prediction of the object
            success, bbox = tracker.update(next_image.copy())
            if pred_counter >= N_FRAMES:
                success = False
            if success:
                pred_counter += 1
                xmin, ymin, w, h = map(int, bbox)
                xmax = xmin + w
                ymax = ymin + h
                obj = [gClassIdx, xmin, ymin, xmax, ymax]
                frame_data_dict = json_file_add_object(frame_data_dict, frame_path, anchor_id, pred_counter, obj)
                cv2.rectangle(next_image, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
                # save prediction
                annotation_paths = get_annotation_paths(frame_path, annotation_formats)
                save_bounding_box(annotation_paths, gClassIdx, (xmin, ymin), (xmax, ymax), self.img_w, self.img_h)
                # show prediction
                cv2.imshow(WINDOW_NAME, next_image)
                pressed_key = cv2.waitKey(DELAY)
            else:
                break

        json_file_data.update({'n_anchor_ids': (anchor_id + 1)})
        # save the updated data
        with open(json_file_path, 'w') as outfile:
            json.dump(json_file_data, outfile, sort_keys=True, indent=4)


# Backup current annotations and then clear them
def clear_bboxes():
    global gImgObjects, gImgIdx
    img_objects_bak = gImgObjects.copy()

    img_path = get_img_path()
    for path in get_annotation_paths(img_path, annotation_formats):
        if os.path.exists( path ):
            os.remove( path )
    set_img_index( gImgIdx )
    return img_objects_bak


def restore_bboxes( img, annotation_paths, img_objects ):
    height, width = img.shape[:2]
    for class_index, x1, y1, x2, y2 in img_objects:
        save_bounding_box(annotation_paths, class_index, (x1,y1), (x2,y2), width, height)


def run_yolo( img, yolo ):
    results = yolo.runInference( img )
    yolo_img_objs = []
    for result in results:
        classIdx = int(result[3])
        x,y,w,h = yolo2xywh( result, img.shape )
        yolo_img_objs.append( ( classIdx, x, y, x+w, y+h ) )
    return yolo_img_objs


def complement_bgr(color):
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)


def get_dominant_mask( img, thresh=.05 ):
    dominant_color = get_dominant_color( img )
    mask = cv2.inRange( img, dominant_color * ( 1 - thresh ), dominant_color * ( 1 + thresh ) )
    return mask


def get_dominant_color( img ):
    reshaped = img.reshape(-1,3)
    kmeans = MiniBatchKMeans( n_clusters=3 ).fit(reshaped)
    bins = np.bincount(kmeans.labels_)
    dominant_idx = bins.argmax()
    dominant_color = kmeans.cluster_centers_[dominant_idx].astype(np.int)
    return dominant_color

def auto_contour_threshold( img, stop_avg_mult_thresh = 1.05, start_test_val = 30, low_thresh = 5 ):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    this_change_perc = 0
    change_sum = 0
    change_cnt = 0

    prev_cnt = None
    ret_thresh = None

    prev_test_val = start_test_val
    for test_val in range( start_test_val, low_thresh, -2 ):
        edge_img = cv2.Canny( imgray, low_thresh, test_val )
        _, contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if prev_cnt is not None:
            # How many contours are there, as a % of the previous value's count
            this_change_perc = len(contours)/prev_cnt
            change_sum += this_change_perc
            change_cnt += 1

            # How does this current % compare to the average %?
            avg_change = change_sum / change_cnt
            avg_mult = this_change_perc / avg_change

            #print( f"{test_val} - Change is {this_change_perc}.  Avg {avg_change}   Mult: {avg_mult}")
            # If the change greater than is 'stop_avg_mult_thresh' times the average then stop here
            if avg_mult > stop_avg_mult_thresh:
                ret_thresh = prev_test_val
                break
        prev_cnt = len(contours)
        prev_test_val = test_val

    return ret_thresh


def draw_masked( img, mask_thresh=.05):
    mask = get_dominant_mask( img, mask_thresh )
    masked = cv2.bitwise_and( img, img, mask=mask )
    return masked

def draw_contours( img, thresh_low, thresh_high ):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.Canny( imgray, thresh_low, thresh_high )
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_img = cv2.drawContours(img, [contour], 0, (255,255,0), 1)
    return contour_img

# change to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # load all images and videos (with multiple extensions) from a directory using OpenCV
    IMAGE_PATH_LIST = []
    VIDEO_NAME_DICT = {}

    if args.yoloWeights:
        try:
            from trackerTools.yoloInference import YoloInference
            yolo = YoloInference( args.yoloWeights )
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            yolo = None
    else:
        yolo = None

    for root, dirs, files in os.walk(INPUT_DIR):
        for f in sorted(files, key = natural_sort_key):
            f_path = os.path.join(root, f)
            f_ext = os.path.splitext( f_path.lower() )[1]
            if os.path.isdir(f_path):
                # skip directories
                continue
            # check if it is an image
            if f_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                IMAGE_PATH_LIST.append(f_path)
            elif f_ext in ['.czi']:
                desired_img_format = '.png'
                converted_path = convert_czi_to_image( f_path, desired_img_format )
                IMAGE_PATH_LIST.append(converted_path)
            else:
                # test if it is a video
                test_video_cap = cv2.VideoCapture(f_path)
                n_frames = int(test_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_video_cap.release()
                if n_frames > 0:
                    # it is a video
                    desired_img_format = '.jpg'
                    video_frames_path, video_name_ext = convert_video_to_images(f_path, n_frames, desired_img_format)
                    # add video frames to image list
                    frame_list = sorted(os.listdir(video_frames_path), key = natural_sort_key)
                    ## store information about those frames
                    first_index = len(IMAGE_PATH_LIST)
                    last_index = first_index + len(frame_list) # exclusive
                    indexes_dict = {}
                    indexes_dict['first_index'] = first_index
                    indexes_dict['last_index'] = last_index
                    VIDEO_NAME_DICT[video_name_ext] = indexes_dict
                    IMAGE_PATH_LIST.extend((os.path.join(video_frames_path, frame) for frame in frame_list))
    last_img_index = len(IMAGE_PATH_LIST) - 1

    # create output directories
    if len(VIDEO_NAME_DICT) > 0:
        if not os.path.exists(TRACKER_DIR):
            os.makedirs(TRACKER_DIR)
    for ann_dir in annotation_formats:
        new_dir = os.path.join(OUTPUT_DIR, ann_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for video_name_ext in VIDEO_NAME_DICT:
            new_video_dir = os.path.join(new_dir, video_name_ext)
            if not os.path.exists(new_video_dir):
                os.makedirs(new_video_dir)


    # load class list
    with open('class_list.txt') as f:
        CLASS_LIST = list(nonblank_lines(f))
    #print(CLASS_LIST)
    last_class_index = len(CLASS_LIST) - 1

    # Make the class colors the same each session
    # The colors are in BGR order because we're using OpenCV
    class_rgb = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (0, 192, 255), (255, 192, 0), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
    class_rgb = np.array(class_rgb)
    # If there are still more classes, add new colors randomly
    num_colors_missing = len(CLASS_LIST) - len(class_rgb)
    if num_colors_missing > 0:
        more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
        class_rgb = np.vstack([class_rgb, more_colors])

    # create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

    cv2.namedWindow(ZOOM_WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(ZOOM_WINDOW_NAME, 400, 400)

    # selected image
    cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, set_img_index)

    # selected class
    if last_class_index != 0:
        cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, set_class_index)

    # initialize
    set_img_index(0)
    edges_on = False
    text_on = True
    show_only_active_class = False
    img_obj_bak = []
    base_img = None
    gRedrawNeeded = True
    invert_image = False
    masked_on = False

    contours_on = False
    contour_thresh_low = 6
    contour_thresh_high = 14
    contour_kmean_thresh = .05
    global thresh_high_offset, thresh_low_offset
    thresh_high_offset = 0
    thresh_low_offset = 0
    zoom_radius = ZOOM_RADIUS
    play_video = False

    display_text('Welcome!\n Press [h] for help.', 4000)

    # loop
    while True:
        color = class_rgb[gClassIdx].tolist()

        # Draw the image except the mouse
        if gRedrawNeeded:
            gRedrawNeeded = False
            base_img = gOrigImg.copy()
            height, width = base_img.shape[:2]
            if edges_on == True:
                # draw edges
                base_img = draw_edges(base_img)

            if contours_on == True:
                contour_thresh_high = auto_contour_threshold( base_img, low_thresh = contour_thresh_low )
                base_img = draw_contours( base_img, contour_thresh_low, contour_thresh_high )

            if invert_image:
                base_img = cv2.bitwise_not(base_img)

            if masked_on == True:
                base_img = draw_masked(base_img, mask_thresh=contour_kmean_thresh )

            # draw already done bounding boxes
            if not show_only_active_class:
                class_filter = None
            else:
                class_filter = gClassIdx

            # get annotation paths
            img_path = get_img_path()
            annotation_paths = get_annotation_paths(img_path, annotation_formats)
            base_img = draw_bboxes_from_file(base_img, annotation_paths, class_filter)

        tmp_img = base_img.copy()
        # draw vertical and horizontal guide lines
        draw_line(tmp_img, gMouseX, gMouseY, height, width, color)
        # write selected class
        class_name = CLASS_LIST[gClassIdx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        margin = 3
        text_width, text_height = cv2.getTextSize(class_name, font, font_scale, LINE_THICKNESS)[0]
        if text_on:
            tmp_img = cv2.rectangle(tmp_img, (gMouseX + LINE_THICKNESS, gMouseY - LINE_THICKNESS), (gMouseX + text_width + margin, gMouseY - text_height - margin), complement_bgr(color), -1)
            tmp_img = cv2.putText(tmp_img, class_name, (gMouseX + margin, gMouseY - margin), font, font_scale, color, LINE_THICKNESS, cv2.LINE_AA)


        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_mouse_move(gMouseX, gMouseY)

        # if bounding box is selected add extra info
        if gIsBboxSelected:
            tmp_img = draw_info_bb_selected(tmp_img)
        # if first click
        if gPoint1[0] != -1:
            # draw partial bbox
            cv2.rectangle(tmp_img, gPoint1, (gMouseX, gMouseY), color, LINE_THICKNESS)
            # if second click
            if gPoint2[0] != -1:
                # save the bounding box
                save_bounding_box(annotation_paths, gClassIdx, gPoint1, gPoint2, width, height)
                # reset the points
                gPoint1 = (-1, -1)
                gPoint2 = (-1, -1)

        cv2.imshow(WINDOW_NAME, tmp_img)

        img_y, img_x, _ = tmp_img.shape
        crop_left = gMouseX - zoom_radius
        crop_right = gMouseX + zoom_radius
        if crop_left < 0:
            crop_left = 0
            crop_right = min( 2*zoom_radius, img_x )
        elif crop_right > img_x:
            crop_right = img_x
            crop_left = max( crop_right - 2*zoom_radius, 0 )

        crop_top = gMouseY - zoom_radius
        crop_bottom = gMouseY + zoom_radius
        if crop_top < 0:
            crop_top = 0
            crop_bottom = min( 2*zoom_radius, img_y )
        elif crop_bottom > img_y:
            crop_bottom = img_y
            crop_top = max( crop_bottom - 2*zoom_radius, 0)
        cv2.imshow(ZOOM_WINDOW_NAME, tmp_img[crop_top:crop_bottom, crop_left:crop_right])

        pressed_key = cv2.waitKey(DELAY)

        # Play video until the end of the video is reached or any key is pressed
        if play_video:
            if -1 != pressed_key:
                play_video = False
                pressed_key = -1
            else:
                endIdx = get_vid_img_index( gImgIdx, 1, last_img_index )
                nextIdx = increase_index(gImgIdx, last_img_index)
                if nextIdx != endIdx:
                    cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, nextIdx)
                else:
                    play_video = False

        if dragBBox.anchor_being_dragged is None:
            ''' Key Listeners START '''
            if pressed_key == ord('A') or pressed_key == ord('a') or pressed_key == ord('D') or pressed_key == ord('d'):
                # show previous image key listener
                if pressed_key == ord('a'):
                    gImgIdx = decrease_index(gImgIdx, last_img_index)
                elif pressed_key == ord('A'):
                    gImgIdx = get_vid_img_index( gImgIdx, -1, last_img_index )
                # show next image key listener
                elif pressed_key == ord('d'):
                    gImgIdx = increase_index(gImgIdx, last_img_index)
                elif pressed_key == ord('D'):
                    gImgIdx = get_vid_img_index( gImgIdx, 1, last_img_index )
                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, gImgIdx)
                img_obj_bak = []
            elif pressed_key == ord(' '):
                play_video = True
            elif pressed_key == ord('r'):
                gRedrawNeeded = True
            elif pressed_key == ord('i'):
                invert_image = not invert_image
                gRedrawNeeded = True
            elif pressed_key == ord('1'):
                zoom_radius += ZOOM_STEP
            elif pressed_key == ord('3'):
                zoom_radius = max( ZOOM_MIN, zoom_radius - ZOOM_STEP )
            elif pressed_key == ord('g'):
                contours_on = not contours_on
                gRedrawNeeded = True
            elif pressed_key == ord('m'):
                masked_on = not masked_on
                gRedrawNeeded = True
            elif pressed_key == ord('s') or pressed_key == ord('w'):
                # change down current class key listener
                if pressed_key == ord('s'):
                    gClassIdx = decrease_index(gClassIdx, last_class_index)
                # change up current class key listener
                elif pressed_key == ord('w'):
                    gClassIdx = increase_index(gClassIdx, last_class_index)
                draw_line(tmp_img, gMouseX, gMouseY, height, width, color)
                set_class_index(gClassIdx)
                cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, gClassIdx)
                if gIsBboxSelected:
                    matched_obj = gImgObjects[gSelectedBbox]
                    edit_bbox(matched_obj, 'change_class:{}'.format(gClassIdx))
            # help key listener
            elif pressed_key == ord('h'):
                text = ('[e] to show edges;\n'
                        '[q] to quit;\n'
                        '[a] or [d] to change Image;\n'
                        '[A] or [D] to change Video;\n'
                        '[w] or [s] to change Class.\n'
                        '[t] to toggle text\n'
                        '[c] to toggle inactive showing only active class bboxes\n'
                        '[y] to clear bboxes and run yolo inference\n'
                        '[1|3] zoom in|zoom out\n'
                        '[p] Track objects from the previous frame into this frame\n'
                        '[P] Track objects (just selected, or all if none selected) from this frame into the rest of video\n'
                        '[0] Delete the selected object, from this frame and the rest of the video\n'
                        '[space] Auto-advance through the current video. Press any key to stop\n'
                        )
                display_text(text, 5000)
            # show edges key listener
            elif pressed_key == ord('e'):
                edges_on = not edges_on
                display_text( f"Edges turned {'ON' if edges_on else 'OFF'}!", 1000)
                gRedrawNeeded = True
            elif pressed_key == ord('t'):
                text_on = not text_on
                display_text( f"Text turned {'ON' if text_on else 'OFF'}!", 1000)
                gRedrawNeeded = True
            elif pressed_key == ord('c'):
                show_only_active_class = not show_only_active_class
                display_text( f"Show only active class bboxes turned {'ON' if show_only_active_class else 'OFF'}!", 1000)
                gRedrawNeeded = True
            elif pressed_key == ord('y') and yolo is not None:
                if len(img_obj_bak) == 0 :
                    img_obj_bak = clear_bboxes()
                    gImgObjects = run_yolo( gOrigImg, yolo )
                    restore_bboxes( tmp_img, annotation_paths, gImgObjects )
            elif pressed_key == ord('u'):
                tmp = gImgObjects.copy()
                restore_bboxes( tmp_img, annotation_paths, img_obj_bak)
                img_obj_bak = tmp
            elif pressed_key == ord('P') or pressed_key == ord('p') or pressed_key == ord('0'):
                singleFrame = ( pressed_key == ord('p') )
                deleteBbox = ( pressed_key == ord('0') )
                selected_bbox = None
                # check if the image is a frame from a video
                is_from_video, video_name = is_frame_from_video(img_path)
                if is_from_video and ( gSelectedBbox != -1 or deleteBbox == False ):
                    # get list of objects associated to that frame
                    object_list = np.array(gImgObjects)
                    origImgIdx = gImgIdx

                    object_tracker = ObjectTracker()

                    # Grab objects from previous frame if just single-frame
                    if singleFrame:
                        gImgIdx = decrease_index(gImgIdx, last_img_index)
                        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, gImgIdx)
                        imgPath = get_img_path()
                        object_list = np.array(gImgObjects)
                    elif gIsBboxSelected:
                        # Track an object, but delete it each frame
                        selected_bbox = object_list[gSelectedBbox]
                        selected_bbox_class = selected_bbox[0]
                        object_list = np.array([selected_bbox])
                        if deleteBbox:
                            edit_bbox(selected_bbox, 'delete')


                    if len(object_list) == 0:
                        print(f"Can't track video objects when no objects are marked in current frame")
                    else:
                        # Initialize trackers with the current frames annotated bounding boxes
                        object_tracker.setImage( gOrigImg )
                        object_classes = object_list[:,0]
                        object_bboxes = object_list[:,1:]
                        object_bboxes = [ x1y1x2y22rxywh( bbox, gOrigImg.shape ) for bbox in object_bboxes ]
                        object_metadata = [ { "class": objClass } for objClass in object_classes ]
                        object_tracker.updateDetections( object_bboxes, object_metadata)

                    objects = object_tracker.update()
                    if len(objects) > 0:
                        # Get list of frames
                        next_frame_path_list = get_next_frame_path_list(video_name, img_path)
                        last_idx = get_vid_img_index( gImgIdx, 1, last_img_index )
                        exitLoop = False
                        # Iterate through all of the remaining video frames
                        while gImgIdx != last_idx and not exitLoop:
                            # Get the next image
                            gImgIdx = increase_index(gImgIdx, last_img_index)
                            cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, gImgIdx)
                            imgPath = get_img_path()
                            annotation_paths = get_annotation_paths(imgPath, annotation_formats)

                            # Update the trackers with the new image
                            img = gOrigImg.copy()
                            object_tracker.setImage( gOrigImg )
                            objects = object_tracker.update()

                            if len(gImgObjects) > 0:
                                # Deal with existing annotations
                                annotated_bboxes = [ x1y1x2y22rxywh( obj[1:], img.shape ) for obj in gImgObjects]
                                annotated_classes = [ obj[0] for obj in gImgObjects]
                                updatedBoxes, matchedIds, unmatchedIds = object_tracker.matchDetections( annotated_bboxes, annotated_classes )

                                # Initially don't add any tracked objects
                                trackedIdsToAdd = { key: False for key in objects.keys() }

                                for trackerId in unmatchedIds:
                                    # We tracked an object that is not annotated in this scene, so add it
                                    trackedIdsToAdd[trackerId] = True

                                # TODO: Revisit this loop. Do we need to run it if !selected_bbox?
                                for idx,trackerId in enumerate(matchedIds):
                                    if trackerId is not None:
                                        if selected_bbox is not None:
                                            # This must match our selected box
                                            matched_obj = gImgObjects[idx]
                                            matched_obj_class = matched_obj[0]
                                            if matched_obj_class == selected_bbox_class:
                                                if deleteBbox:
                                                    edit_bbox(matched_obj, 'delete')
                                                else: # Auto-adding a single tracked object, and it already is in frame?
                                                    print(f"Found a matching object in new frame, stopping auto-add.")
                                                    exitLoop = True
                                            else:
                                                if deleteBbox:
                                                    print(f"Class mismatch in deleting matching object. Looking for {selected_bbox_class}, found {matched_obj_class}")
                                                    exitLoop = True
                                                else:
                                                    # The closest match isn't the same object, so we need to add our tracked object
                                                    assert( len(updatedBoxes) == 1 )
                                                    trackedIdsToAdd[trackedId] = True
                                            break
                                    else:
                                        pass
                            else:
                                # No annotations? Then add every tracked object
                                trackedIdsToAdd = { key: True for key in objects.keys() }

                            for trackerId,shouldAdd in trackedIdsToAdd.items():
                                if shouldAdd:
                                    tracker = object_tracker._trackers[trackerId]
                                    x,y,w,h = rxywh2x1y1wh(tracker.lastSeen, img.shape)
                                    imgHeight, imgWidth = img.shape[:2]
                                    classId = int( tracker.metadata['class'] )
                                    save_bounding_box( annotation_paths, classId, (x,y), (x+w,y+h), imgWidth, imgHeight )

                            dbgImg = img.copy()
                            draw_bboxes_from_file( dbgImg, annotation_paths )
                            cv2.imshow( WINDOW_NAME, dbgImg )
                            if cv2.waitKey(25) != -1:
                                break

                            if singleFrame:
                                break

                            # for key,tracker in object_tracker._trackers.items():
                            #     x1,y1,w,h = rxywh2x1y1wh(tracker.lastSeen, img.shape)
                            #     cv2.rectangle( img, (x1,y1), (x1+w,y1+h), (0,0,255))
                            #     cv2.putText( img, f"Id:{key} Missed: {tracker.lostCount}", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                            # cv2.imshow( WINDOW_NAME, img )
                            # cv2.waitKey()

                    gRedrawNeeded = True

                    # If we were deleting a bbox we should return to the original frame
                    if deleteBbox:
                        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, origImgIdx)


                    # # remove the objects in that frame that are already in the `.json` file
                    # json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
                    # file_exists, json_file_data = get_json_file_data(json_file_path)
                    # if file_exists:
                    #     object_list = remove_already_tracked_objects(object_list, img_path, json_file_data)
                    # if len(object_list) > 0:
                    #     # get list of frames following this image
                    #     next_frame_path_list = get_next_frame_path_list(video_name, img_path)
                    #     # initial frame
                    #     init_frame = gOrigImg.copy()
                    #     label_tracker = LabelTracker(TRACKER_TYPE, init_frame, next_frame_path_list)
                    #     for obj in object_list:
                    #         gClassIdx = obj[0]
                    #         color = class_rgb[gClassIdx].tolist()
                    #         label_tracker.start_tracker(json_file_data, json_file_path, img_path, obj, color, annotation_formats)
            # quit key listener
            elif pressed_key == ord('q'):
                break
            ''' Key Listeners END '''

        if WITH_QT:
            # if window gets closed then quit
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
