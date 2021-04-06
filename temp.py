import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
video_path = 'max.avi'
pixeis = []
cap = cv.VideoCapture(video_path)
while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        frame = frame [350:354, 350:354,:] 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pixeis.append(gray)
    except:
        break

p1 = []
amplified = []
tempo = np.linspace(0, 764, num=764, endpoint=True, dtype=int)
for i in range(len(tempo)):
    p1.append(int(pixeis[i][0][0]))
p1 = np.array(p1)

for i in range(1,len(tempo)):

    if(p1[i]-p1[i-1] == 0 and not i==1):
        amplified.append(amplified[-1])
    else:
        if(p1[i]<p1[i-1] and not i==1):
            amplified.append(p1[i-1]-(p1[i]-p1[i-1])**2)
        else:
            amplified.append(p1[i-1]+(p1[i]-p1[i-1])**2)
amplified.append(amplified[-1])
plt.figure()
plt.subplot(121)
plt.plot(tempo, p1)
plt.subplot(122)
plt.plot(tempo, amplified)
cap.release()
