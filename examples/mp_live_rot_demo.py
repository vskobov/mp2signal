#!/usr/bin/env python
"""
Mediapipe To Signal Tool. 
Converts mediapipe holistic pose estimation into normalized 3d body series of joint's angles representation.
Normalizes the pose base on body proportions. 

Copyright (C) 2021-2023, Victor Skobov 
All rights reserved. E-mail: <vskobov@gmail.com>.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
from cv2 import COLOR_BGR2RGB
from matplotlib.pyplot import text
import mediapipe as mp
import numpy as np
import mp2signal.mp2s as mp2s

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def get_rot(h,w):
    ret = np.zeros((h,w,3))
    ret.fill(0)
    return ret

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

PAUSED = False

frame_counter = 0
b_mul = None
ref_face = True
with mp_holistic.Holistic(
    smooth_landmarks=True,
    model_complexity=2,
    min_detection_confidence=0.1,
    refine_face_landmarks=ref_face,
    min_tracking_confidence=0.1) as holistic:

    while cap.isOpened():
        if PAUSED==False:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue


            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            image.flags.writeable = False
            results = holistic.process(image)

            X_DIMENSION = 600   
            Y_DIMENSION = 600
            

            before_image = np.zeros((X_DIMENSION, Y_DIMENSION,3),dtype=np.uint8)
            after_image = np.zeros((X_DIMENSION, Y_DIMENSION,3),dtype=np.uint8)
            #face_image = np.zeros((X_DIMENSION, Y_DIMENSION,3),dtype=np.uint8)

            if results:
                if ref_face:
                    MP_Face = np.zeros((1,478,4))
                else:
                    MP_Face = np.zeros((1,468,4))

                MP_Pose = np.zeros((1,33,7))
                MP_RHand = np.zeros((1,21,4))
                MP_LHand = np.zeros((1,21,4))
                j_list=[]
                j_list = [2,5,3,4,6,7,
                            811,999,
                            1124,1053,1065,1055,#right brow
                            1285,1295,1276,1353,#left brow
                            1033,1160,1157,1133,1153,1144,#right eye
                            1263,1387,1385,1362,1381,1373,#left eye
                            1078,1012,1308,1015#mouth
                            ]
                
                l_hand=[700+k for k in range(21)]
                r_hand=[400+k for k in range(21)]
                j_list = j_list+l_hand+r_hand
                j_list=[]
                if results.face_landmarks:
                    MP_Face[0] = mp2s.mp_frame_coords(results.face_landmarks,image_height,image_width)
                if results.pose_landmarks and results.pose_world_landmarks:
                    MP_Pose[0] = mp2s.mp_frame_coords(results.pose_landmarks,image_height,image_width,results.pose_world_landmarks)

                if results.right_hand_landmarks:
                    MP_RHand[0]= mp2s.mp_frame_coords(results.right_hand_landmarks,image_height,image_width)
                if results.left_hand_landmarks:
                    MP_LHand[0]= mp2s.mp_frame_coords(results.left_hand_landmarks,image_height,image_width)

                
                if results.pose_landmarks:
                    mt = mp2s.Joint_Tree(mp2s.COCO_BODY_TREE)
                    res_movement = {'MP_Pose':MP_Pose,
                                'MP_Face':MP_Face,
                                'MP_RHand':MP_RHand,
                                'MP_LHand':MP_LHand}

                    mt.process(res_movement,b_mul=b_mul)

                    before_image = mt._draw_tree(before_image,0,False,j_list,text=["Original",'Frame:',frame_counter,'B_mul',int(mt.body_muls[0])])
                    after_image = mt._draw_tree(after_image,0,True,j_list,text=["Rotated",'Frame:',frame_counter,'B_mul',int(mt.body_muls[0])],add_face=True)
                    #face_image = mt._draw_face(face_image,0,[],text=["Rotated",'Frame:',frame_counter,'B_mul',int(mt.body_muls[0])])
                    b_mul = int(mt.body_muls[0])

            else: 
                text = 'No one is detetected'
                cv2.putText(after_image,str(text), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            side_by_side = np.concatenate((before_image,after_image),axis=1)
            cv2.imshow('Rotation MP2S', cv2.cvtColor(side_by_side,COLOR_BGR2RGB))
            frame_counter +=  1
            wk = cv2.waitKey(5)
            if wk & 0xFF == ord('p'):
                PAUSED =True

        else:
            wk = cv2.waitKey(5)
            if wk & 0xFF == ord('p'):
                PAUSED =False

        if wk & 0xFF == 27:
            break

cap.release()
