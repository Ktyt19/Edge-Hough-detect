import cv2 # type: ignore
import numpy as np # type: ignore

# โหลดภาพ
image_color = cv2.imread('circle2.png')
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
image= cv2.GaussianBlur(image, (3, 3), 1)
image = cv2.Canny(image, 50, 200)

# image: ภาพที่ต้องการตรวจจับวงกลม
# cv2.HOUGH_GRADIENT: วิธีการคำนวณ Hough Transform สำหรับวงกลม
# dp=1: สัดส่วนของความละเอียดในการตรวจจับวงกลม (1 หมายถึงความละเอียดเดิม)
# minDist=20: ระยะห่างขั้นต่ำระหว่างจุดศูนย์กลางของวงกลมที่ตรวจจับ
# param1=50: ค่าขอบเขตของ Canny edge detector
# param2=30: ค่าขอบเขตสำหรับการตรวจจับวงกลม
# minRadius=0: รัศมีของวงกลมที่น้อยที่สุด
# maxRadius=0: รัศมีของวงกลมที่มากที่สุด (0 หมายถึงไม่จำกัด)
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                           param1=50, param2=30, minRadius=0, maxRadius=0)


# วาดวงกลมลงบนภาพเฉพาะ
sum=0
if circles is not None:
   circles = np.round(circles[0, :]).astype("int")
   for (x, y, r) in circles:
      cv2.circle(image_color, (x, y), r, (0, 255 ,0 ), 2)  # เปลี่ยนสีเป็นสีแดง
      sum+=1

print(sum)
# แสดงภาพที่มีวงกลมตรวจจับแล้ว
cv2.imshow("Detected Circles", image_color)
cv2.imshow("Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
