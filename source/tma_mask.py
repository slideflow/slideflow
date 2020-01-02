import cv2
import numpy as np
import openslide as ops
import math
import sys
import matplotlib.pylab as plt

from os.path import join
from statistics import mean, median

DOWNSCALE = 100
WIDTH_MIN = 20
HEIGHT_MIN = 20

save_loc = "C:\\Users\James\\Documents\\Research\\TMA\\tiles"

tma_full = "C:\\Users\\James\\Documents\\Research\\TMA\\TMA_1185.svs"
slide = ops.OpenSlide(tma_full)
MPP = float(slide.properties[ops.PROPERTY_NAME_MPP_X])
DIM = slide.dimensions

img_orig = np.array(slide.get_thumbnail((DIM[0]/100, DIM[1]/100)))
img_annotated = img_orig.copy()
img_h, img_w, img_c = img_annotated.shape

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def getSubImage(rect, src):
    box = cv2.boxPoints(rect) * DOWNSCALE
    box = np.int0(box)
    rect_width = int(rect[1][0] * DOWNSCALE)
    rect_height = int(rect[1][1] * DOWNSCALE)

    region_x_min = int(min([b[0] for b in box]))
    region_x_max = max([b[0] for b in box])
    region_y_min = int(min([b[1] for b in box]))
    region_y_max = max([b[1] for b in box])

    region_width = region_x_max - region_x_min
    region_height = region_y_max - region_y_min

    extracted = slide.read_region((region_x_min, region_y_min), 0, (region_width, region_height))
    relative_box = box - [region_x_min, region_y_min]

    print(relative_box[0])
    print(box[0])
    print(region_x_min)
    print('---')

    src_pts = relative_box.astype("float32")
    dst_pts = np.array([[0, rect_height-1],
                        [0, 0],
                        [rect_width-1, 0],
                        [rect_width-1, rect_height-1]], dtype="float32")
    P = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped=cv2.warpPerspective(np.array(extracted), P, (rect_width, rect_height))
    return warped

print(f"Full slide microns-per-pixel: {MPP}")
print(f"Full slide dimensions: {DIM}")

inc = np.array([255,255,255])
bound = 28
upper = inc - bound
lower = np.array([0,0,0])

#cv2.imshow("Original", img)
mask = cv2.inRange(img_orig, np.array([0,0,0]), upper)
#cv2.imshow("Range", mask)

closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
dilating_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
closing = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, closing_kernel)
dilated = cv2.dilate(closing, dilating_kernel)

contours, heirarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

box_areas = []
img_cropped = 0

for i, component in enumerate(zip(contours, heirarchy[0])):
    cnt = component[0]
    heir = component[1]
    rect = cv2.minAreaRect(cnt)
    width = rect[1][0]
    height = rect[1][1]
    if width > WIDTH_MIN and height > HEIGHT_MIN and heir[3] < 0:
        moment = cv2.moments(cnt)
        cX = int(moment["m10"] / moment["m00"])
        cY = int(moment["m01"] / moment["m00"])
        cv2.drawContours(img_annotated, contours, i, (255,0,0))
        cv2.circle(img_annotated, (cX, cY), 7, (150, 255, 150), -1)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = PolyArea([b[0] for b in box], [b[1] for b in box])
        box_areas += [area]
        cv2.drawContours(img_annotated, [box], 0, (200, 100, 100), 2)
        img_cropped += 1   
        cv2.putText(img_annotated, f'{img_cropped}', (cX+10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 230), 2)
    elif heir[3] < 0:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_annotated, [box], 0, (100, 200, 100), 2)

print(f"Number of regions: {len(contours)}")
print(f"Number of filtered regions: {img_cropped}")

cv2.imshow("Annotated", cv2.resize(img_annotated, (1400, 1000)))
cv2.waitKey(0)

img_cropped = 0
for i, component in enumerate(zip(contours, heirarchy[0])):
    cnt = component[0]
    heir = component[1]
    rect = cv2.minAreaRect(cnt)
    width = rect[1][0]
    height = rect[1][1]
    if width > WIDTH_MIN and height > HEIGHT_MIN and heir[3] < 0:
        img_crop = getSubImage(rect, img_orig)
        img_cropped += 1
        cv2.imwrite(join(save_loc, f"tile{img_cropped}.jpg"), img_crop)

print("Summary of box areas:")
print(f"Min: {min(box_areas) * DOWNSCALE * MPP:.1f}")
print(f"Max: {max(box_areas) * DOWNSCALE * MPP:.1f}")
print(f"Mean: {mean(box_areas) * DOWNSCALE * MPP:.1f}")
print(f"Median: {median(box_areas) * DOWNSCALE * MPP:.1f}")

#cv2.imshow("Original", cv2.resize(img_orig, (1400, 1000)))
#cv2.imshow("Dilated mask", cv2.resize(dilated, (800, 600)))