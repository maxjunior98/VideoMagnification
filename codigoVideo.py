import cv2 as cv
import logging
import random
import string
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from os.path import isfile, join


def lenght_of_video(pathVideo):          #funcao para pegar lenght do video
    cap = cv.VideoCapture(pathVideo)
    lenght = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return lenght

def size_x_matrix(matrix):           #funcao para pegar largura do video
    columns = len(matrix[0])
    return int(columns)

def size_y_matrix(matrix):           #funcao para pegar altura do video
    rows = len(matrix)
    return int(rows)

def get_fps(pathVideo):
    cam = cv.VideoCapture(pathVideo)
    fps = cam.get(cv.CAP_PROP_FPS)
    return fps

def convert_to_video(pathOut, fps, frames_array, height, width, len_video):
    size = (width, height)
    out = cv.VideoWriter(pathOut, cv.VideoWriter_fourcc(*'DIVX'), fps,size)
    for k in range(len_video):
        out.write(frames_array[k])
    out.release()


path = 'max.avi'
lenpath = lenght_of_video(path)     #lenght do video
fpsVideo = get_fps(path)            #frames per second do video
pathExit = './results/test1.avi'        

pixels = []         #LISTA DE MATRIZ COM OS VALORES DOS PIXELS DE CADA FRAME
cap = cv.VideoCapture(path)
while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        frame = frame [300:700,700:1200,:]              #PEGANDO PARTE DO VIDEO
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    #CONVERTE PARA GRAY
        pixels.append(gray)
    except:
        break

size_largura = size_x_matrix(pixels[0])     #pegar altura e largura do video
size_altura = size_y_matrix(pixels[0])     

#p1 = []
#amplified = []
#tempo = np.linspace(0, lenpath, num=lenpath, endpoint=True, dtype=int)

#ANALISE GRÁFICA DO COMPORTAMENTO DE UM PIXEL PARA DEFINIR FATOR DE AMPLIACAO
# for i in range(len(tempo)):
#     p1.append(int(pixels[i][300][400]))
# p1 = np.array(p1)

# for x in range(len(tempo)):
#     if(x == 0):
#         amplified.append(p1[x])
#     else:
#         if(p1[x] == p1[x-1]):
#             amplified.append(amplified[x - 1])
#         else:
#             print('FRAME')
#             print(x)
#             if(p1[x] > p1[x - 1]):
#                 print('fator')
#                 delta = p1[x] - p1[x-1]
#                 print(delta)
#                 print('fator multiplicado')
#                 delta = 5 * delta
#                 print(delta)
#                 if((amplified[x-1] + delta)>255):
#                     amplified.append(255)
#                 else:
#                     amplified.append(amplified[x-1] + delta)
#             else:
#                 print('fator')
#                 delta = p1[x-1] - p1[x]
#                 print(delta)
#                 print('fator multiplicado')
#                 delta = 5 * delta
#                 print(delta)
#                 if((amplified[x-1] - delta)<0):
#                     amplified.append(0)
#                 else:
#                     amplified.append(amplified[x-1] - delta)

# plt.figure()
# plt.subplot(121)
# plt.plot(tempo, p1)
# plt.subplot(122)
# plt.plot(tempo, amplified)

Matrix_amplified = []
for index in range(lenpath):
    Matrix_amplified.append(pixels[index].copy())

for index_height in range(size_altura):
    for index_width in range(size_largura):
        for index_frame in range(180, 500):
            value = int(pixels[index_frame][index_height][index_width])
            prevalue = int(pixels[index_frame - 1][index_height][index_width])
            # print('PIXEL:')
            # print(index_width)
            # print(index_height)
            # print('FRAME')
            # print(index_frame)
            # print('Valor')
            # print(value)
            # print('Valor Anterior')
            # print(prevalue)
            if(value != prevalue):
                # print('DIF')
                dif = value - prevalue
                # print(dif)
                dif = 2*dif
                # print('DIF AUMENTADA')
                # print(dif)
                if(dif > 0):
                    value =  value + dif
                    if(value > 255):
                        value = 255
                    Matrix_amplified[index_frame][index_height][index_width] = value
                else:
                    value = value + dif
                    if(value < 0):
                        value = 0
                    Matrix_amplified[index_frame][index_height][index_width] = value
            else:
                Matrix_amplified[index_frame][index_height][index_width] = Matrix_amplified[index_frame - 1][index_height][index_width]
         

    
input("Press Enter to continue...")

for mmmm in range(180, 500):
    cv.imshow('figure 2' ,Matrix_amplified[mmmm])
    k = cv.waitKey(20)
    if(k == 33):
        break

input("Press Enter to continue...")

video_exit = pixels
for mmmm in range(lenpath):
    cv.imshow('figure 2' ,video_exit[mmmm])
    k = cv.waitKey(20)
    if(k == 33):
        break

cap.release()
#convert_to_video(pathExit, fpsVideo, video_exit, size_altura, size_largura, lenpath)