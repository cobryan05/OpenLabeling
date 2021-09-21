#!/bin/python
from __future__ import annotations

import argparse
import copy
import os
import pathlib
import re
import sys

import cv2
import numpy as np
import czifile

from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

submodules_dir = os.path.join( pathlib.Path('__file__').parent.resolve(), "..", "submodules" )
sys.path.append( submodules_dir )
sys.path.append( os.path.join(submodules_dir, "yolov5") )

from trackerTools.objectTracker import ObjectTracker
from trackerTools.bbox import BBox
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


class TaggedObject:
    def __init__(self):
        self.bbox : BBox = None
        self.classIdx : int = -1
        self.name : str = None
        self.trackerId : int = -1
        self.fixed : bool = False

    def __eq__(self, other):
        return other is not None and self.bbox == other.bbox

    @staticmethod
    def fromYoloLine( yoloLine: str ) -> TaggedObject:
        newObject = TaggedObject()
        parts = yoloLine.split(' ')
        newObject.classIdx = int(parts[0])
        newObject.name = CLASS_LIST[newObject.classIdx]
        newObject.bbox = BBox.fromYolo( *map(float, parts[1:]) )
        return newObject


    def yoloLine(self):
        yolo_info = map(str, [ self.classIdx, *self.bbox.asYolo() ] )
        return ' '.join(yolo_info)


class TaggedObjectManager:
    def __init__(self):
        self._tracker : ObjectTracker = None
        self._trackedObjects : dict[int, TaggedObject] = {}
        self._objectList : list[TaggedObject]  = []
        self._selectedIdx : int = -1

    def SelectNext(self) -> TaggedObject:
        if self._selectedIdx < len(self._objectList) - 1:
            self._selectedIdx += 1
        else:
            self._selectedIdx = 0

    def SelectPrev(self) -> TaggedObject:
        if self._selectedIdx > 0:
            self._selectedIdx = min( self._selectedIdx - 1, len(self._objectList) - 1 )
        else:
            self._selectedIdx = len(self._objectList) - 1

    @property
    def objectList( self ):
        return self._objectList

    @objectList.setter
    def objectList(self, objects : list[TaggedObject] ):
        prevSelObj = self.selectedObject
        self._objectList = objects.copy()
        self.selectedObject = None
        if prevSelObj:
            similars = self.getSimilarObjects( prevSelObj.bbox, epsilon=.1 )
            if len(similars) == 1:
                self.selectedObject = similars[0]


    @property
    def selectedObject( self ):
        if self._selectedIdx == -1:
            return None
        return self.objectList[self._selectedIdx]

    @selectedObject.setter
    def selectedObject( self, obj : TaggedObject ):
        if obj is None:
            self._selectedIdx = -1
        else:
            try:
                self._selectedIdx = self.objectList.index(obj)
            except ValueError:
                print(f"Failed to set selected item to {obj}")
                self.selectedObject = None


    def getSimilarObjects( self, bbox : BBox, epsilon = BBox.EPSILON ) -> list[TaggedObject]:
        return [ obj for obj in self.objectList if obj.bbox.similar(bbox, epsilon ) ]

    def initObjectTracker( self, img : np.ndarray, trackerType:str = "CSRT" ):
        self._tracker = ObjectTracker( trackerType )
        self._trackedObjects = {}
        self._tracker.setImage( img )
        for obj in self.objectList:
            obj.trackerId = self._tracker.addObject( obj.bbox )
            self._trackedObjects[obj.trackerId] = obj


    def trackNewImage( self, img : np.ndarray, trackByClass : bool = True ):
        if self._tracker is None:
            return

        # Set the new image and update the tracker
        self._tracker.setImage( img )
        trackerResults = self._tracker.update()

        # First try to match up existing boxes
        for obj in self.objectList:
            for id,trackerBbox in trackerResults.items():
                if trackerBbox is None:
                    continue
                if trackerBbox.similar(obj.bbox, epsilon=.2):
                    obj.trackerId = id
                    trackerResults.pop(id)
                    break

        # Do another pass, matching unmatched boxes by classIdx
        if trackByClass:
            idsToRemove = set()
            for id,trackerBbox in trackerResults.items():
                prevTrackedObj = self._trackedObjects.get(id, None)
                if prevTrackedObj:
                    # Attempt to find an untracked object by class before creating a new one
                    for obj in self.objectList:
                        if prevTrackedObj.classIdx == obj.classIdx and obj.trackerId == -1: # TODO: We don't initialize the id
                            obj.trackerId = id
                            idsToRemove.add(id)
                            break
            for id in idsToRemove:
                trackerResults.pop(id)

        # Now add any new boxes tracked in
        idsToRemove = set()
        for id,trackerBbox in trackerResults.items():
            for id,trackerBbox in trackerResults.items():
                prevTrackedObj = self._trackedObjects.get(id, None)
                if prevTrackedObj:
                    if not prevTrackedObj.fixed:
                        prevTrackedObj.bbox = trackerBbox
                    self.objectList.append(prevTrackedObj)
                    idsToRemove.add(id)
        for id in idsToRemove:
            trackerResults.pop(id)

        if len(trackerResults) > 0:
            print(f"There were {len(trackerResults)} unhandled tracker results!")


gObjManager : TaggedObjectManager = TaggedObjectManager()

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

ANNOTATION_FORMATS = {'YOLO_darknet' : '.txt'}
TRACKER_DIR = os.path.join(OUTPUT_DIR, '.tracker')

DRAW_FROM_PASCAL = args.draw_from_PASCAL_files

# selected bounding box
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
    sRA = LINE_THICKNESS * 4

    # Object being dragged
    selected_object: TaggedObject = None

    # Flag indicating which resizing-anchor is dragged
    anchor_being_dragged = None

    '''
    \brief This method is used to check if a current mouse position is inside one of the resizing anchors of a bbox
    '''
    @staticmethod
    def check_point_inside_resizing_anchors(eX, eY, obj: TaggedObject):
        global gOrigImg
        imgY, imgX = gOrigImg.shape[:2]
        x_left, y_top, x_right, y_bottom = obj.bbox.asX1Y1X2Y2( imgX, imgY )
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
    def handler_left_mouse_down(eX, eY, obj : TaggedObject):
        dragBBox.check_point_inside_resizing_anchors(eX, eY, obj)
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.selected_object = obj

    @staticmethod
    def handler_mouse_move(eX, eY):
        global gRedrawNeeded, gOrigImg
        imgY, imgX = gOrigImg.shape[:2]
        if dragBBox.selected_object is not None:
            x_left, y_top, x_right, y_bottom = dragBBox.selected_object.bbox.asX1Y1X2Y2(imgX, imgY)

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
    global gImgIdx, gOrigImg, gObjManager, gRedrawNeeded
    gImgIdx = x
    img_path = get_img_path()
    gOrigImg = cv2.imread(img_path)
    gRedrawNeeded = True
    text = 'Showing image {}/{}, path: {}'.format(str(gImgIdx), str(last_img_index), img_path)
    display_text(text, 1000)

    # create empty annotation files for each image, if it doesn't exist already
    abs_path = os.path.abspath(img_path)
    folder_name = os.path.dirname(img_path)
    image_name = os.path.basename(img_path)
    img_height, img_width, depth = (str(number) for number in gOrigImg.shape)

    annotation_paths = get_annotation_paths(img_path, ANNOTATION_FORMATS)
    gObjManager.objectList = read_objects_from_file( annotation_paths )
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
    global gObjManager
    gObjManager.objectList = read_objects_from_file( annotation_paths )
    return draw_bboxes( tmp_img, gObjManager.objectList, class_filter = class_filter)

def read_objects_from_file( annotation_paths ) -> list[TaggedObject]:
    global gOrigImg
    fileObjects : list[TaggedObject] = []

    ann_path = None
    if DRAW_FROM_PASCAL:
        # Read objects from PASCAL file
        ann_path = next(path for path in annotation_paths if 'PASCAL_VOC' in path)
    else:
        # Read objects from YOLO file
        ann_path = next(path for path in annotation_paths if 'YOLO_darknet' in path)

    imgY, imgX = gOrigImg.shape[:2]
    if os.path.isfile(ann_path):
        if DRAW_FROM_PASCAL:
            tree = ET.parse(ann_path)
            annotation = tree.getroot()
            for idx, obj in enumerate(annotation.findall('object')):
                class_name, class_index, xmin, ymin, xmax, ymax = get_xml_object_data(obj)
                #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                newObject = TaggedObject()
                newObject.bbox = BBox.fromX1Y1X2Y2( xmin, ymin, xmax, ymax, imgX, imgY )
                newObject.classIdx = class_index
                fileObjects.append(newObject)
        else:
            # Draw from YOLO
            with open(ann_path) as fp:
                for idx, line in enumerate(fp):
                    newObject = TaggedObject.fromYoloLine(line)
                    fileObjects.append(newObject)
    return fileObjects


def draw_bboxes( img, objects: list[TaggedObject], class_filter = None ):
    global gObjManager, gOrigImg
    imgY, imgX = gOrigImg.shape[:2]
    for obj in objects:
        class_idx = obj.classIdx
        x1, y1, x2, y2 = obj.bbox.asX1Y1X2Y2( imgX, imgY )
        color = class_rgb[class_idx].tolist()

        # draw resizing anchors if the object is selected
        if obj == gObjManager.selectedObject:
            img = draw_bbox_anchors(img, x1, y1, x2, y2, color)

        # draw bbox
        if class_filter is None or class_filter == class_idx:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_THICKNESS)
            if text_on:
                font = cv2.FONT_HERSHEY_SIMPLEX
                label = f"{obj.name}"
                if obj.trackerId != -1:
                    label = f"Id: {obj.trackerId} - {label}"
                if obj.fixed:
                    label = f"[F] {label}"
                cv2.putText(img, label, (x1, y1 - 5), font, 0.6, color, LINE_THICKNESS, cv2.LINE_AA)
    return img



def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height


def select_bbox_under_mouse(set_class):
    global gObjManager, gOrigImg

    imgY, imgX = gOrigImg.shape[:2]
    smallest_area = -1
    selected_obj : TaggedObject = None
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(gObjManager.objectList):
        x1, y1, x2, y2 = obj.bbox.asX1Y1X2Y2( imgX, imgY )
        x1 = x1 - dragBBox.sRA
        y1 = y1 - dragBBox.sRA
        x2 = x2 + dragBBox.sRA
        y2 = y2 + dragBBox.sRA
        if pointInRect(gMouseX, gMouseY, x1, y1, x2, y2):
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_obj = obj

    if selected_obj is not None:
        set_selected_object( selected_obj, set_class )

def set_selected_object(obj: TaggedObject, select_class = False):
    global gObjManager, gRedrawNeeded
    gObjManager.selectedObject = obj

    if select_class:
        # set class to the one of the selected bounding box
        cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, obj.classIdx)
    gRedrawNeeded = True



def edit_bbox(obj_to_edit: TaggedObject, action):
    ''' action = `delete`
                 `change_class:new_class_index`
                 `resize_bbox:new_x_left:new_y_top:new_x_right:new_y_bottom`
    '''
    global gOrigImg, gObjManager, gRedrawNeeded
    imgY, imgX = gOrigImg.shape[:2]

    orig_obj = copy.copy(obj_to_edit)

    if 'change_class' in action:
        obj_to_edit.classIdx = int(action.split(':')[1])
        obj_to_edit.name = CLASS_LIST[obj_to_edit.classIdx]
    elif 'resize_bbox' in action:
        new_x_left = max(0, int(action.split(':')[1]))
        new_y_top = max(0, int(action.split(':')[2]))
        new_x_right = min(imgX, int(action.split(':')[3]))
        new_y_bottom = min(imgY, int(action.split(':')[4]))
        obj_to_edit.bbox = BBox.fromX1Y1X2Y2( new_x_left, new_y_top, new_x_right, new_y_bottom, imgX, imgY )
    elif 'delete' in action:
        obj_to_edit = None
    elif 'add' in action:
        pass
    else:
        print(f"Unknown edit_bbox action: {action}")
        return

    current_img_path = get_img_path()

    for ann_path in get_annotation_paths(current_img_path, ANNOTATION_FORMATS):
        if '.txt' in ann_path:
            # edit YOLO file
            with open(ann_path, 'r') as old_file:
                lines = old_file.readlines()

            with open(ann_path, 'w') as new_file:
                if 'add' in action:
                    new_file.write( obj_to_edit.yoloLine() + '\n' )

                for line in lines:
                    if not orig_obj.bbox.similar( TaggedObject.fromYoloLine( line ).bbox ):
                        # If not the object to edit then just copy it
                        new_file.write(line)
                    elif obj_to_edit:
                        new_yolo_line = obj_to_edit.yoloLine()
                        new_file.write(new_yolo_line + '\n')

        elif '.xml' in ann_path:
            assert(False) # PASCAL VOC is unmantained
            # edit PASCAL VOC file
            tree = ET.parse(ann_path)
            annotation = tree.getroot()
            class_index = orig_obj.classIdx
            xmin, ymin, xmax, ymax = orig_obj.bbox.asX1Y1X2Y2(imgX, imgY)
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
                        object_class.text = CLASS_LIST[obj_to_edit.classIdx]
                    elif 'resize_bbox' in action:
                        object_bbox = obj.find('bndbox')
                        object_bbox.find('xmin').text = str(new_x_left)
                        object_bbox.find('ymin').text = str(new_y_top)
                        object_bbox.find('xmax').text = str(new_x_right)
                        object_bbox.find('ymax').text = str(new_y_bottom)
                    break

            xml_str = ET.tostring(annotation)
            write_xml(xml_str, ann_path)

    # Update the manager if we deleted an object
    if 'delete' in action:
        if orig_obj == gObjManager.selectedObject:
            gObjManager.selectedObject = None
        gObjManager.objectList.remove( orig_obj )

    gRedrawNeeded = True


def mouse_listener(event, x, y, flags, param):
    # mouse callback function
    global gObjManager, gMouseX, gMouseY, gPoint1, gPoint2, gRedrawNeeded, gOrigImg

    if event == cv2.EVENT_MOUSEMOVE:
        gMouseX = x
        gMouseY = y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        #print('Double click')
        gPoint1 = (-1, -1)
        # if clicked inside a bounding box we set that bbox
        select_bbox_under_mouse(True)
    # By AlexeyGy: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        select_bbox_under_mouse(False)
        if gObjManager.selectedObject:
            edit_bbox(gObjManager.selectedObject, 'delete')
    # Change class to current class via middle-click
    elif event == cv2.EVENT_MBUTTONDOWN:
        select_bbox_under_mouse(False)
        if gObjManager.selectedObject:
            edit_bbox(gObjManager.selectedObject, f'change_class:{gClassIdx}')

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Check if mouse inside on of resizing anchors of the selected bbox
        if gObjManager.selectedObject:
            dragBBox.handler_left_mouse_down(x, y, gObjManager.selectedObject)

        if dragBBox.anchor_being_dragged is None:
            if gPoint1[0] == -1:
                if gObjManager.selectedObject:
                    gObjManager.selectedObject = None
                    gRedrawNeeded = True
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
    # for idx, obj in enumerate(gImgObjects):
        # ind, x1, y1, x2, y2 = obj
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


def save_bounding_box(annotation_paths, bbox : BBox, class_index ):
    global gOrigImg
    for ann_path in annotation_paths:
        if '.txt' in ann_path:
            tmp = TaggedObject()
            tmp.classIdx = class_index
            tmp.bbox = bbox
            line = tmp.yoloLine()
            append_bb(ann_path, line, '.txt')
        elif '.xml' in ann_path:
            imgY, imgX = gOrigImg.shape[:2]
            line = voc_format(CLASS_LIST[class_index], *bbox.asX1Y1X2Y2( imgX, imgY ) )
            append_bb(ann_path, line, '.xml')

def is_frame_from_video(img_path):
    img_path_parts = os.path.normpath(img_path).split(os.sep)
    for video_name in VIDEO_NAME_DICT:
        if video_name in img_path_parts:
            # image belongs to a video
            return True, video_name
    return False, None


def get_cropped_img( img, centerX, centerY, width, height ):
    img_y, img_x, _ = img.shape
    crop_left = centerX - width
    crop_right = centerX + width
    if crop_left < 0:
        crop_left = 0
        crop_right = min( 2*width, img_x )
    elif crop_right > img_x:
        crop_right = img_x
        crop_left = max( crop_right - 2*width, 0 )

    crop_top = centerY - height
    crop_bottom = centerY + height
    if crop_top < 0:
        crop_top = 0
        crop_bottom = min( 2*height, img_y )
    elif crop_bottom > img_y:
        crop_bottom = img_y
        crop_top = max( crop_bottom - 2*height, 0)
    return img[crop_top:crop_bottom, crop_left:crop_right]


def remove_annotation_file():
    img_path = get_img_path()
    for path in get_annotation_paths(img_path, ANNOTATION_FORMATS):
        if os.path.exists( path ):
            os.remove( path )


def restore_bboxes( img, img_objects: list[TaggedObject] ):
    img_path = get_img_path()
    annotation_paths = get_annotation_paths(img_path, ANNOTATION_FORMATS)
    for obj in img_objects:
        save_bounding_box(annotation_paths, obj.bbox, obj.classIdx)


def run_yolo( img, yolo ):
    results = yolo.runInference( img )
    yolo_img_objs = []
    for (x,y),(w,h),conf,objclass in results:
        yolo_img_objs.append( ( objclass, x, y, x+w, y+h ) )
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


def run_tracker( selectedObj : TaggedObject, singleFrame : bool, deleteInFrames: bool):
    global gImgIdx, gObjManager, gOrigImg, gMouseX, gMouseY

    object_list = gObjManager.objectList

    if singleFrame:
        # Scan back frames to find annotations if no object selected
        if selectedObj is None:
            singleFrameIdx = gImgIdx
            # Scan back for a frame that has annotations
            while True:
                gImgIdx = decrease_index(gImgIdx, last_img_index)
                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, gImgIdx)
                imgPath = get_img_path()
                object_list = gObjManager.objectList
                if len(object_list) > 0 or singleFrameIdx == gImgIdx:
                    break
        else:
            # Track the selected object forward into the next frame
            pass
    elif selectedObj:
        # Track an object, but delete it each frame
        object_list = [selectedObj]
        if deleteBbox:
            edit_bbox(object_list[0], 'delete')

    curSelObj = selectedObj
    if len(object_list) == 0:
        print(f"Can't track video objects when no objects are marked in current frame")
    else:
        # Initialize trackers with the current frames annotated bounding boxes
        tmpObjMan = TaggedObjectManager()
        tmpObjMan.objectList = object_list
        tmpObjMan.initObjectTracker( gOrigImg )

        # Get list of frames
        vid_last_idx = get_vid_img_index( gImgIdx, 1, last_img_index )
        exitLoop = False

        # If only processing a single frame we may have rewound quite a bit before finding an annotated frame.
        #  Put us back to 1 before our active frame
        if singleFrame and curSelObj is None:
            gImgIdx = decrease_index(singleFrameIdx, last_img_index )

        # Iterate through all of the remaining video frames
        while gImgIdx != vid_last_idx and not exitLoop:
            # Get the next image
            gImgIdx = increase_index(gImgIdx, last_img_index)
            cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, gImgIdx)
            tmpObjMan.objectList = gObjManager.objectList # Update tracker manager with new image annotations

            # Update the trackers with the new image
            tmpObjMan.trackNewImage( gOrigImg )

            # If a selected object is tracked then try to update it to current image objects
            newSelectedObj = None
            if curSelObj:
                similarObjs = gObjManager.getSimilarObjects( curSelObj.bbox, epsilon=.1 )
                if len(similarObjs) == 1:
                    newSelectedObj = similarObjs[0]
            curSelObj = newSelectedObj


            if deleteInFrames:
                # If deleting and there is a similar object, delete it. Otherwise break the loop
                if curSelObj:
                    edit_bbox( curSelObj, 'delete' )
                else:
                    exitLoop = True
            else:
                if curSelObj:
                    exitLoop = True # This object is already in this frame
                else:
                    # Update the image with all tracked boxes
                    for trackedObj in tmpObjMan.objectList:
                        similarObjs = gObjManager.getSimilarObjects( trackedObj.bbox )
                        if not len(similarObjs):
                            # Add new object
                            gObjManager.objectList.append(trackedObj)
                            edit_bbox( trackedObj, 'add' )

                            # Track forward - TODO: Not so hacky
                            if selectedObj and singleFrame:
                                gObjManager.selectedObject = trackedObj
                                exitLoop = True
            # Show image
            dbgImg = gOrigImg.copy()
            draw_bboxes( dbgImg, gObjManager.objectList )
            cv2.imshow( WINDOW_NAME, dbgImg )
            zoomed_dbg_img = get_cropped_img( dbgImg, gMouseX, gMouseY, zoom_radius, zoom_radius )
            cv2.imshow(ZOOM_WINDOW_NAME, zoomed_dbg_img)
            if cv2.waitKey(1) != -1:
                exitLoop = True

            if singleFrame:
                exitLoop = True



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
    for ann_dir in ANNOTATION_FORMATS:
        new_dir = os.path.join(OUTPUT_DIR, ann_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

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
            annotation_paths = get_annotation_paths(img_path, ANNOTATION_FORMATS)
            base_img = draw_bboxes( base_img, gObjManager.objectList, class_filter)

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
        if gObjManager.selectedObject:
            tmp_img = draw_info_bb_selected(tmp_img)
        # if first click
        if gPoint1[0] != -1:
            # draw partial bbox
            cv2.rectangle(tmp_img, gPoint1, (gMouseX, gMouseY), color, LINE_THICKNESS)
            # if second click
            if gPoint2[0] != -1:
                # save the bounding box
                bbox = BBox.fromX1Y1X2Y2( *gPoint1, *gPoint2, width, height)

                newObject = TaggedObject()
                newObject.bbox = bbox
                newObject.classIdx = gClassIdx
                newObject.name = CLASS_LIST[gClassIdx]
                gObjManager.objectList.append(newObject)
                save_bounding_box(annotation_paths, bbox, gClassIdx)
                # reset the points
                gPoint1 = (-1, -1)
                gPoint2 = (-1, -1)

        cv2.imshow(WINDOW_NAME, tmp_img)
        zoomed_img = get_cropped_img( tmp_img, gMouseX, gMouseY, zoom_radius, zoom_radius )
        cv2.imshow(ZOOM_WINDOW_NAME, zoomed_img)

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

        if pressed_key != -1 and dragBBox.anchor_being_dragged is None:
            ''' Key Listeners START '''
            if pressed_key in [ ord('['), ord('{'), ord(']'), ord('}') ]:
                # show previous image key listener
                if pressed_key == ord('['):
                    gImgIdx = decrease_index(gImgIdx, last_img_index)
                elif pressed_key == ord('{'):
                    gImgIdx = get_vid_img_index( gImgIdx, -1, last_img_index )
                # show next image key listener
                elif pressed_key == ord(']'):
                    gImgIdx = increase_index(gImgIdx, last_img_index)
                elif pressed_key == ord('}'):
                    gImgIdx = get_vid_img_index( gImgIdx, 1, last_img_index )
                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, gImgIdx)
            elif pressed_key == ord(' '):
                play_video = True
            elif pressed_key == ord('r'):
                gRedrawNeeded = True
            elif pressed_key == ord('f'):
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
            elif pressed_key == ord('o'):
                gObjManager.initObjectTracker( gOrigImg, "CSRT")
                gRedrawNeeded = True
            elif pressed_key == ord('O'):
                gObjManager.trackNewImage( gOrigImg )
                remove_annotation_file()
                restore_bboxes( gOrigImg, gObjManager.objectList)
                gRedrawNeeded = True
            elif pressed_key in [ord('e'), ord('q')]:
                # change down current class key listener
                if pressed_key == ord('e'):
                    gClassIdx = increase_index(gClassIdx, last_class_index)
                # change up current class key listener
                elif pressed_key == ord('q'):
                    gClassIdx = decrease_index(gClassIdx, last_class_index)
                set_class_index(gClassIdx)
                cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, gClassIdx)
                if gObjManager.selectedObject:
                    edit_bbox(gObjManager.selectedObject, 'change_class:{}'.format(gClassIdx))
            # help key listener
            elif pressed_key == ord('h'):
                text = ('[z] to show edges;\n'
                        '[ESC] to quit;\n'
                        '[[] or []] to change Image;\n'
                        '[{] or [}] to change Video;\n'
                        '[q] or [e] to change Class.\n'
                        '[f] to invert image\n'
                        '[w,a,s,d] move bbox side in\n'
                        '[W,A,S,D] move bbox side out\n'
                        '[t] to toggle text\n'
                        '[c] to toggle inactive showing only active class bboxes\n'
                        '[y] to clear bboxes and run yolo inference\n'
                        '[1|3] zoom in|zoom out\n'
                        '[p] Track objects from the previous frame into this frame\n'
                        '[P] Track objects (just selected, or all if none selected) from this frame into the rest of video\n'
                        '[0] Delete the selected object, from this frame and the rest of the video\n'
                        '[`] or [~] cycle selected bbox'
                        '[space] Auto-advance through the current video. Press any key to stop\n'
                        )
                display_text(text, 5000)
            # show edges key listener
            elif pressed_key == ord('z'):
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
            elif gObjManager.selectedObject and pressed_key == ord('='):
                gObjManager.selectedObject.fixed = not gObjManager.selectedObject.fixed
                gRedrawNeeded = True
            elif gObjManager.selectedObject and pressed_key in [ ord('a'), ord('A'), ord('w'), ord('W'), ord('d'), ord('D'), ord('s'), ord('S') ]:
                imgY, imgX = gOrigImg.shape[:2]
                left, top, right, bottom = gObjManager.selectedObject.bbox.asX1Y1X2Y2(imgX, imgY)
                offset = 3
                if pressed_key == ord('a'):
                    left -= offset
                elif pressed_key == ord('A'):
                    left += offset
                elif pressed_key == ord('w'):
                    top -= offset
                elif pressed_key == ord('W'):
                    top += offset
                elif pressed_key == ord('d'):
                    right += offset
                elif pressed_key == ord('D'):
                    right -= offset
                elif pressed_key == ord('s'):
                    bottom += offset
                elif pressed_key == ord('S'):
                    bottom -= offset

                action = "resize_bbox:{}:{}:{}:{}".format(left, top, right, bottom)
                edit_bbox(gObjManager.selectedObject, action)
                gRedrawNeeded = True

            elif pressed_key in [ord('`'), ord('~')]:
                if pressed_key == ord('`'):
                    gObjManager.SelectPrev()
                elif pressed_key == ord('~'):
                    gObjManager.SelectNext()
                gRedrawNeeded = True
            elif pressed_key == ord('y') and yolo is not None:
                remove_annotation_file()
                yoloDetections = run_yolo( gOrigImg, yolo )
                imgY, imgX = gOrigImg.shape[:2]
                yoloObjects: list[TaggedObject] = []
                for det in yoloDetections:
                    newObject = TaggedObject()
                    newObject.classIdx = det[0]
                    newObject.bbox = BBox.fromX1Y1X2Y2( *det[1:], imgX, imgY )
                    newObject.name = CLASS_LIST[newObject.classIdx]

                    yoloObjects.append(newObject)

                gObjManager.objectList = yoloObjects
                restore_bboxes( tmp_img, gObjManager.objectList )
            elif pressed_key == ord('P') or pressed_key == ord('p') or pressed_key == ord('0'):
                singleFrame = ( pressed_key == ord('p') )
                deleteBbox = ( pressed_key == ord('0') )
                selectedObj = gObjManager.selectedObject
                # check if the image is a frame from a video
                is_from_video, video_name = is_frame_from_video(img_path)
                if is_from_video and ( selectedObj or deleteBbox == False ):
                    origImgIdx = gImgIdx
                    run_tracker( selectedObj, singleFrame, deleteBbox )

                    gRedrawNeeded = True

                    # If we were deleting a bbox we should return to the original frame
                    if deleteBbox:
                        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, origImgIdx)


            # quit key listener
            elif pressed_key == 27: # ESC on Windows
                break
            ''' Key Listeners END '''

        if WITH_QT:
            # if window gets closed then quit
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
