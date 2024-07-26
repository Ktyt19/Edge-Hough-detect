import cv2 # type: ignore
import numpy as np # type: ignore

img = cv2.imread('road1.jpg', cv2.IMREAD_GRAYSCALE)
canny_edges = cv2.Canny(img, 50, 150)

# ใช้ Probabilistic Hough Line Transform เพื่อค้นหาเส้นตรง
# พารามิเตอร์:
# - canny_edges: ภาพไบนารีที่ได้จากการตรวจจับขอบ
# - 1: ความละเอียดของ ρ (ระยะห่างในหน่วยพิกเซล)
# - np.pi / 180: ความละเอียดของ θ (มุมในหน่วยเรเดียน, เท่ากับ 1 องศา)
# - 50: เกณฑ์การลงคะแนน (จำนวนขั้นต่ำของการลงคะแนนสำหรับการตรวจจับเส้น)
# - minLineLength=50: ความยาวขั้นต่ำของเส้นตรงที่ต้องการตรวจจับ (ในพิกเซล)
# - maxLineGap=10: ช่องว่างสูงสุดระหว่างเซกเมนต์ของเส้นตรงที่สามารถรวมเป็นเส้นเดียวได้ (ในพิกเซล)
lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

# แปลงภาพขาวดำเป็นภาพสี (RGB) เพื่อวาดเส้นตรง
img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(lines)

if lines is not None:
   sum=0
   for line in lines:
      x1, y1, x2, y2 = line[0]
      # ภาพต้นฉบับ: ภาพที่เราต้องการวาดเส้นบนภาพนี้.
      # พิกัด (x1, y1): จุดเริ่มต้นของเส้นตรงในภาพ.
      # พิกัด (x2, y2): จุดสิ้นสุดของเส้นตรงในภาพ.
      # สีแดง: เส้นตรงจะเป็นสีแดง.
      # ความหนา 1 พิกเซล: เส้นตรงจะมีความหนา 1 พิกเซล.
      cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
      sum+=1

print(sum)
cv2.imshow("Hough Lines P", img_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
