import cv2 # type: ignore
import numpy as np # type: ignore
from scipy import ndimage # type: ignore

# # สร้าง kernel สำหรับ Roberts Cross Edge Detection
# kernelGaus = np.array([[1*(1/15), 2*(1/15), 1*(1/15)],
#                      [2*(1/15), 3*(1/15), 2*(1/15)],
#                      [1*(1/15), 2*(1/15), 1*(1/15)]])

# อ่านภาพเข้ามา
img = cv2.imread('image4.png', cv2.IMREAD_GRAYSCALE)

if img is not None:
   # GaussianBlur(img, (ขนาด 3*3), ค่าsigma )
   blur = cv2.GaussianBlur(img, (3, 3), 1)
   # Canny(part , threshold Low, threshold High)
   edged_img = cv2.Canny(blur,0,255)

    # แสดงภาพ binary ที่ได้
   cv2.imshow("Edge image", edged_img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
else:
   print("Error")
