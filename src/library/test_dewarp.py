import cv2
import numpy as np

# read image
img = cv2.imread("/home/app/file/input/Deskew_1.jpg")
hh, ww = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 25
)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

# get convex hull
points = np.column_stack(np.where(morph.transpose() > 0))
hull = cv2.convexHull(points)
peri = cv2.arcLength(hull, True)
hullimg = img.copy()
hullimg = cv2.polylines(hullimg, [hull], True, (0, 255, 0), 1)

# get 4 corners
poly = cv2.approxPolyDP(hull, 0.01 * peri, True)
plist = poly.tolist()
print(plist)
# plist ordered cw from bottom right
cornerimg = img.copy()
cv2.polylines(cornerimg, [np.int32(poly)], True, (0, 0, 255), 1, cv2.LINE_AA)

# specify control point pairs
# order pts cw from bottom right
inpts = np.float32(poly)
outpts = np.float32([[ww, hh], [0, hh], [0, 0], [ww, 0]])

# warp image
M = cv2.getPerspectiveTransform(inpts, outpts)
warpimg = cv2.warpPerspective(img, M, (ww, hh))

cv2.imwrite("thresh.png", thresh)
cv2.imwrite("morph.png", morph)
cv2.imwrite("hullimg.png", hullimg)
cv2.imwrite("cornerimg.png", cornerimg)
cv2.imwrite("warpimg.png", warpimg)
