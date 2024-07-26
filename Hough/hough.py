import cv2  # type: ignore
import numpy as np  # type: ignore
from scipy import ndimage  # type: ignore

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

def houghLineP(edge,img):
# ใช้ Probabilistic Hough Line Transform เพื่อค้นหาเส้นตรง
# พารามิเตอร์:
# - canny_edges: ภาพไบนารีที่ได้จากการตรวจจับขอบ
# - 1: ความละเอียดของ ρ (ระยะห่างในหน่วยพิกเซล)
# - np.pi / 180: ความละเอียดของ θ (มุมในหน่วยเรเดียน, เท่ากับ 1 องศา)
# - 50: เกณฑ์การลงคะแนน (จำนวนขั้นต่ำของการลงคะแนนสำหรับการตรวจจับเส้น)
# - minLineLength=50: ความยาวขั้นต่ำของเส้นตรงที่ต้องการตรวจจับ (ในพิกเซล)
# - maxLineGap=10: ช่องว่างสูงสุดระหว่างเซกเมนต์ของเส้นตรงที่สามารถรวมเป็นเส้นเดียวได้ (ในพิกเซล)
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

# แปลงภาพขาวดำเป็นภาพสี (RGB) เพื่อวาดเส้นตรง
    img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print(lines)
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
    return img_lines,sum

def houghCircles(edge,img): 
# image: ภาพที่ต้องการตรวจจับวงกลม
# cv2.HOUGH_GRADIENT: วิธีการคำนวณ Hough Transform สำหรับวงกลม
# dp=1: สัดส่วนของความละเอียดในการตรวจจับวงกลม (1 หมายถึงความละเอียดเดิม)
# minDist=20: ระยะห่างขั้นต่ำระหว่างจุดศูนย์กลางของวงกลมที่ตรวจจับ
# param1=50: ค่าขอบเขตของ Canny edge detector
# param2=30: ค่าขอบเขตสำหรับการตรวจจับวงกลม
# minRadius=0: รัศมีของวงกลมที่น้อยที่สุด
# maxRadius=0: รัศมีของวงกลมที่มากที่สุด (0 หมายถึงไม่จำกัด)
    circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                           param1=50, param2=30, minRadius=0, maxRadius=0)
    img_circle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    sum=0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img_circle, (x, y), r, (0, 255 ,0 ), 2)  # เปลี่ยนสีเป็นสีแดง
            sum+=1
    return img_circle,sum
    
    
img = cv2.imread('circle3.png', cv2.IMREAD_GRAYSCALE)

print("1.Robert   2.Prewitt")
print("3.Sobel    4.Canny")
choice = input("Input 1-4 : ")
if choice=="1":
    edge=robertEdge(img)
elif choice=="2":
    edge=prewittEdge(img)
elif choice=="3":
    edge=sobelEdge(img)
elif choice=="4":
    edge=cannyEdge(img)

# image,line=houghLineP(edge,img)   
image,line=houghCircles(edge,img)   

print(line)
cv2.imshow("edge", edge)
cv2.imshow("Hough ", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
