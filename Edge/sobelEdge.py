import cv2 # type: ignore
import numpy as np # type: ignore
from scipy import ndimage # type: ignore

# สร้าง kernel สำหรับ Roberts Cross Edge Detection
kernelGx = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])
kernelGy = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1,-2,-1]])

# อ่านภาพเข้ามา
img = cv2.imread('image4.png', cv2.IMREAD_GRAYSCALE)

if img is not None:
    # ทำการ convolution kernel
    gX = ndimage.convolve(img, kernelGx)
    gY = ndimage.convolve(img, kernelGy)

    # คำนวณขอบโดยใช้สูตร sqrt(Gx^2 + Gy^2)
    edged_img = np.sqrt(np.square(gX) + np.square(gY))
    edged_img = np.uint8(edged_img)  # แปลงเป็น np.uint8

    # หา threshold โดยใช้ Otsu's method เพื่อหาค่าที่เหมาะสม
    _, threshold = cv2.threshold(edged_img, 0, 255, cv2.THRESH_OTSU )
    
    # แสดงค่า threshold_value ที่ได้
    # print("Threshold value:", threshold)

    # แสดงภาพ binary ที่ได้
    cv2.imshow("Edge image", threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error")
