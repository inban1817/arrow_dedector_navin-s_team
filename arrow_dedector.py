import cv2 as cv
import numpy as np



def get_arrow_direction(frame):
    r_s=(100,100)
    r_e=(550,300)
    cv.rectangle(frame,r_s,r_e,(0,255,0),2)
    img1=frame[r_s[1]:r_e[1],r_s[0]:r_e[0]]
    img=cv.flip(img1,1)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(gray,(5,5),0)
    r,thresh=cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours,h=cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.imshow("img",img)
    if len(contours)==0:
        return "image not found"
    for c in contours:
        area=cv.contourArea(c)
        if area>1000:
            x,y,w,h=cv.boundingRect(c)
            roi=thresh[y:y+h,x:x+w]
            l_h=np.sum(roi[:,:w//2])
            r_h=np.sum(roi[:,w//2:])
            if l_h>r_h:
                return "left"   
            else:
                return "right"
    return "no arrow found"



#img=cv.imread("right.jpeg")
#img1=cv.flip(img,1)
#print(get_arrow_direction(img1))






cam=cv.VideoCapture(0)

while True:
    r,frame=cam.read()
    result=get_arrow_direction(frame)
    cv.putText(frame,result,(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv.imshow("frame",frame)
    if cv.waitKey(1)==ord("c"):
        break


cv.destroyAllWindows()
cam.release()