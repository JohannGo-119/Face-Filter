import cv2
import numpy as np
import dlib
from math import hypot


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,100)
transformation = cv2.VideoCapture('pics/AttackTransformation1.mp4')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
yes = True
while True:

    success, img = cap.read()
    imgAug = img.copy()

    yes, eren = transformation.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(img)
    for face in faces:
        landmarks = predictor(imgGray, face)
        facecenter = (landmarks.part(30).x, landmarks.part(30).y)
        centerleft = (landmarks.part(0).x, landmarks.part(0).y)
        centerright = (landmarks.part(16).x, landmarks.part(16).y)

        facewidth = int(hypot(centerleft[0]-centerright[0], centerleft[1]-centerright[1]))
        faceheight = int(480*.80)

        myPoints = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])

    #New filter position
        topleft = (int(facecenter[0] - facewidth),
                            int(facecenter[1] - faceheight/2))
        bottomright = (int(facecenter[0] + facewidth),
                            int(facecenter[1] + faceheight/2))

    if yes:
        #adding the filter
        # erencropped = eren[1400:1920, 0:680]
        erenresized = cv2.resize(eren, (facewidth*2, faceheight))
        erencroppedGray = cv2.cvtColor(erenresized, cv2.COLOR_BGR2GRAY)

        success, facemask = cv2.threshold(erencroppedGray, 25, 255, cv2.THRESH_BINARY_INV)

        erenarea = img[0:faceheight,
                   topleft[0]:topleft[0] + facewidth*2]

        imgAug = cv2.bitwise_and(erenarea, erenarea, mask=facemask)
        imgAug = cv2.GaussianBlur(imgAug, (9, 9), 10)
        finalface = cv2.add(imgAug, erenresized)

        img[topleft[1]:topleft[1] +faceheight, topleft[0]:topleft[0] + facewidth * 2] = finalface

        #adding the yellow effects on the webcam#
        myPoints = np.array(myPoints)

        imgColorSur = np.zeros_like(img)
        imgColorSur[:] = 0, 255, 255
        imgColorSur = cv2.bitwise_and(img, imgColorSur)
        # imgColorSur = cv2.GaussianBlur(imgColorSur, (2, 2), 10)
        imgColorSur = cv2.addWeighted(img, 1, imgColorSur, 0.4, 0)
        img = imgColorSur


    if not yes:
        transformation.release() #to not crash the program after video ends
        cv2.destroyWindow('eren')
        eren = cv2.imread('pics/attackTitan.png')
        erencropped = eren[0:1280, 90:590]
        erenarea = img[0:faceheight, topleft[0]:topleft[0] + facewidth * 2]
        erenresized = cv2.resize(erencropped, (facewidth * 2, faceheight))
        erencroppedGray = cv2.cvtColor(erenresized, cv2.COLOR_BGR2GRAY)

        success, facemask = cv2.threshold(erencroppedGray, 25, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow('eren', erenresized)
        # cv2.imshow('yeager', erenarea)

        imgAug = cv2.bitwise_and(erenarea, erenarea, mask=facemask)
        finalface = cv2.add(imgAug, erenresized)

        img[0:  faceheight, topleft[0]:topleft[0] + facewidth * 2] = finalface


    cv2.imshow('frame', img)


    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break