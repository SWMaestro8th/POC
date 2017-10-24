import cv2
import multiprocessing
import time

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import animation
from utils.app_utils import FPS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from moviepy.editor import *
from datetime import datetime


# 모델 설정
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 3


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)



# 객체인식해서 사람의 좌표 반환
# 정확도,속도 향샹 필요
def detect_objects(image_np, w, h, sess, detection_graph):
    startTime = datetime.now()

    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        min_score_thresh=.4,
        use_normalized_coordinates=True,
        line_thickness=8)


    ourTeamPoint = []
    enemyTeamPoint=[]
    otherPoint=[]
    '''

    for detectionObject, detectionScore, detectionBox in zip(classes, scores, boxes):
        for finalObject, finalScore, finalBoxPoint in zip(detectionObject, detectionScore, detectionBox):
            if finalObject == 1 and finalScore > 0.5:

                personXPoint = int(finalBoxPoint[3] * w)
                personYPoint = int(finalBoxPoint[2] * h)
                point = (personXPoint, personYPoint)

                boxX2Point=int(finalBoxPoint[3] * w)
                boxY2Point=int(finalBoxPoint[2] * h)
                boxX1Point=int(finalBoxPoint[1] * w)
                boxY1Point=int(finalBoxPoint[0] * h)

                # 아군 적군 구별
                cutImage = image_np[boxY1Point:int(boxY2Point*.8)+2, boxX1Point:boxX2Point]

                teamCode = teamCutting(cutImage,boxX2Point-boxX1Point,int(boxY2Point*.8)+2-boxY1Point)

               # plt.subplot(121), plt.imshow(cutImage), plt.title(str(teamCode))
               # plt.show()

                # 0은 아군
                if teamCode == 0:
                    ourTeamPoint.append(point)
                # 1은 적군
                elif teamCode == 1:
                    enemyTeamPoint.append(point)
                elif teamCode == -1:
                    otherPoint.append(point)

                print(teamCode)
                '''

    print("trasnfer image : " + str(datetime.now() - startTime))

    # 물체인식 완료된 이미지와 아군/적군 위치 반환
    return image_np, ourTeamPoint, enemyTeamPoint, otherPoint



# 변환 행렬 구하는 함수, 매게변수 : 오리지널 이미지의 변환할 좌표를 입력받음
def getTransMatrix(tl,bl,tr,br):

    # 오리지널 이미지
    pts1 = np.float32([tl, bl, tr, br])

    # 변환된 이미지의 가로세로 길이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    global maxWidth,maxHeight
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # 변환된 새로운 이미지의 가로세로 행렬 만들기
    pts2 = np.array([
        [0, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1]], dtype="float32")

    # 변환 행렬
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return M


# 좌표변환 (아군과 적군)
def transPersonPoint(image_np, ourTeamPoint, enemyTeamPoint, otherPoint, transMatrix):

    transOurTeamPoint=None
    transEnemyTeamPoint=None
    transOtherPoint=None

    dst = cv2.warpPerspective(image_np, transMatrix, (maxWidth, maxHeight))

    # 좌표변환 x 적군 아군 없을 경우
    if ourTeamPoint != []:
        original_1 = np.array([(ourTeamPoint)], dtype=np.float32)
        transOurTeamPoint = cv2.perspectiveTransform(original_1, transMatrix)
        drawCircle(dst, transOurTeamPoint,(255,0,0))


    if enemyTeamPoint != []:
        original_2 = np.array([(enemyTeamPoint)], dtype=np.float32)
        transEnemyTeamPoint = cv2.perspectiveTransform(original_2, transMatrix)
        drawCircle(dst, transEnemyTeamPoint,(255,0,255))

    if otherPoint != []:
        original_3 = np.array([(otherPoint)], dtype=np.float32)
        transOtherPoint = cv2.perspectiveTransform(original_3, transMatrix)
        drawCircle(dst, transOtherPoint,(0,0,255))


    # 점 찍힌 이미지와 변환된 사람 좌표를 리턴함
    return dst,transOurTeamPoint, transEnemyTeamPoint,transOtherPoint


def drawCircle(image, point,rgb):
    for i in point:
        for i2 in i:
            cv2.circle(image, (tuple)(i2), 10, rgb, -1)


def compareRGB(b,g,r,compareRGB):
    count=0
    if b>compareRGB[0][0] and b<compareRGB[1][0]:
        count=count+1
    elif r>compareRGB[0][2] and r<compareRGB[1][2]:
        count = count + 1
    elif g>compareRGB[0][1] and g<compareRGB[1][1]:
        count = count + 1

    if count==3:
        return True

    return False


# 하나의 사람에 대해서 팀 구분하기 (물체인식/상반신짜르기/판단해서 리턴)
def teamCutting(image,maxX,maxY):

    stadiumBGR=[(100,100,100),(150,150,150)]
    ourTeamBGR=[(10,10,10),(50,50,50)]
    enemyTeamBGR=[(80,80,80),(99,99,99)]

    ourCount=0
    enemyCount=0
    otherCount=0

    stadiumCount=0

    for x in range(0,int(maxX)):
        for y in range(0, int(maxY)):

            b = image.item(y, x, 0)
            g = image.item(y, x, 1)
            r = image.item(y, x, 2)

            if compareRGB(b,g,r,stadiumBGR):
                stadiumCount=stadiumCount+1
            else :
                if compareRGB(b,g,r,ourTeamBGR):
                     ourCount=ourCount+1
                elif compareRGB(b,g,r,enemyTeamBGR):
                    enemyCount=enemyCount+1
                else:
                    otherCount=otherCount+1

    if ourCount >= enemyCount:
        return 0
    elif ourCount < enemyCount:
        return 1
    elif otherCount > ourCount and otherCount > enemyCount:
        return -1

def mainProcessing():

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # 버드아이뷰 변환 행렬 구하기
    tl = (480, 494)
    bl = (878,1036 )
    tr = (792, 469)
    br = (1328, 743)
    M = getTransMatrix(tl, bl, tr, br)

    # 비디오 변환
    my_clip = VideoFileClip("/Users/itaegyeong/Desktop/무제 폴더/GOPR0008.MP4")
    w = my_clip.w
    h = my_clip.h

    #plt.ion()
    #fig = plt.figure()
    #sub1 = fig.add_subplot(121)
    #sub2 = fig.add_subplot(122)
    #data = np.zeros((w, h))
    #im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=1)
    #plt.subplot(121), plt.title('image')
    #plt.subplot(122), plt.title('transfer')

    video = cv2.VideoWriter('/Users/itaegyeong/Desktop/tensorflowvideo3.mp4', -1, 30, (w, h))

    a=0
    for frame in my_clip.iter_frames():
        a+=1
        if a%10!=0:
            continue

        # 이미지에서 물체인식하고 아군적군 판단해서 아군의 좌표 적군의 좌표, 인식되 박스쳐진 이미지를 리턴한다
        image, ourTeamPoint, enemyTeamPoint, otherPoint = detect_objects(frame, w, h, sess, detection_graph)

        #이미지와 원래 아군/적군의 좌표, 변환행렬을 매게변수로 전달하여 버드아이뷰로 바꾼다
        # transImage, ourTransTeamPoint, enemyTransTeamPoint, transOtherPoint =transPersonPoint(image,ourTeamPoint,enemyTeamPoint,otherPoint, M)



        video.write(image)
        #image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('asd', image)
        #cv2.imshow('title',transImage)
        #cv2.waitKey(1)


        #plt.subplot(121), plt.imshow(image), plt.title('image')
        #plt.subplot(122), plt.imshow(transImage), plt.title('transfer')
        #im.set_data(transImage)
        if a==300000:
            break

        #fig.canvas.draw()

    cv2.destroyAllWindows()
    video.release()


mainProcessing()
