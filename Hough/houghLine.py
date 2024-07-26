import cv2 # type: ignore
import numpy as np # type: ignore

# อ่านภาพและแปลงเป็นภาพขาวดำ
img = cv2.imread('Test1.png', cv2.IMREAD_GRAYSCALE)

# ตรวจจับขอบในภาพด้วย Canny edge detector
canny_edges = cv2.Canny(img, 0, 255)

# ใช้ Hough Transform ในการตรวจจับเส้นตรง
lines = cv2.HoughLines(canny_edges, 1, np.pi / 180, 100)

# สร้างสำเนาภาพต้นฉบับเพื่อวาดเส้น
img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(lines)
# วาดเส้นที่ตรวจจับได้
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

# แสดงผลลัพธ์
cv2.imshow("Canny Edges", canny_edges)
cv2.imshow("Hough Lines", img_lines)

cv2.waitKey(0)
cv2.destroyAllWindows()
