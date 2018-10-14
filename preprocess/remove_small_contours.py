import cv2
import os
import numpy as np

if __name__ == '__main__':
    img_file = './13.jpg'

    img_src = cv2.imread(img_file)
    gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # ret, gray = cv2.threshold(gray, 250, 255, 0)
    dst_img = np.zeros(img_src.shape, np.uint8)

    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('contour', image)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 32:
            mask = np.ones(gray.shape, np.uint8)*255
            cv2.drawContours(dst_img, cnt, -1, 0, 1)
            img_src = cv2.bitwise_and(img_src, img_src, mask=mask)


            # pass

    # mask = np.zeros(image_src.shape, np.uint8)
    # largest_areas = sorted(contours, key=cv2.contourArea)
    # cv2.drawContours(mask, [largest_areas[-2]], 0, (255, 255, 255, 255), -1)
    # removed = cv2.add(image_src, mask)
    #
    # cv2.imwrite("removed.png", removed)


    cv2.imshow('ori', gray)
    cv2.imshow('dst', img_src)
    cv2.waitKey(0)