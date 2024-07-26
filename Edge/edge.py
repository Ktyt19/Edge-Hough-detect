import cv2 # type: ignore
import numpy as np # type: ignore
from scipy import ndimage # type: ignore

def sobelEdge(img):
    kernelGx = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    kernelGy = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    edged_img = convolution(kernelGx, kernelGy, img)
    _, threshold = cv2.threshold(edged_img, 0, 255, cv2.THRESH_OTSU)
    return threshold

def prewittEdge(img):
    kernelGx = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
    kernelGy = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])
    edged_img = convolution(kernelGx, kernelGy, img)
    _, threshold = cv2.threshold(edged_img, 0, 255, cv2.THRESH_OTSU)
    return threshold

def robertEdge(img):
    kernelGx = np.array([[1, 0], [0, -1]])
    kernelGy = np.array([[0, 1], [-1, 0]])
    edged_img = convolution(kernelGx, kernelGy, img)
    _, threshold = cv2.threshold(edged_img, 0, 255, cv2.THRESH_OTSU)
    return threshold

def cannyEdge(img):
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    edged_img = cv2.Canny(blur, 50, 150)
    return edged_img

def convolution(kernelGx, kernelGy, img):
    gX = ndimage.convolve(img, kernelGx)
    gY = ndimage.convolve(img, kernelGy)
    edged_img = np.sqrt(np.square(gX) + np.square(gY))
    edged_img = np.uint8(edged_img)
    return edged_img

img = cv2.imread('road1.jpg', cv2.IMREAD_GRAYSCALE)

# ตัวอย่างการใช้งานฟังก์ชันการตรวจจับขอบ
sobel_img = sobelEdge(img)
prewitt_img = prewittEdge(img)
robert_img = robertEdge(img)
canny_img = cannyEdge(img)

# สร้างสำเนาภาพต้นฉบับเพื่อวาดเส้น
img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# แสดงการตรวจจับขอบโดยใช้ HoughLines (ตัวอย่าง)
canny_edges = cannyEdge(img)
lines = cv2.HoughLines(canny_edges, 1, np.pi /360 , 150, None, 0, 0)
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imshow("Original Image", img)
# cv2.imshow("Sobel", sobel_img)
# cv2.imshow("Prewitt", prewitt_img)
# cv2.imshow("Robert", robert_img)
cv2.imshow("Canny", canny_img)
cv2.imshow("Hough Lines", img_lines)

cv2.waitKey(0)
cv2.destroyAllWindows()
