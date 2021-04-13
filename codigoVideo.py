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

def get_fps(pathVideo):              #funcao para pegar fps do video
    cam = cv.VideoCapture(pathVideo)
    fps = cam.get(cv.CAP_PROP_FPS)
    return fps

def get_matrix_imgs(path):
    matrix = []
    cap = cv.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame = frame [300:700,700:1200,:]              #PEGANDO PARTE DO VIDEO
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    #CONVERTE PARA GRAY
            matrix.append(gray)
        except:
            break
    cap.release()
    return matrix

def get_frame(matrix, lenpath):
    matrix_exit = []
    for i in range(lenpath):
       matrix_exit.append(int(matrix[i][250][350]))
    matrix_exit = np.array(matrix_exit) 
    return matrix_exit

def get_analise_frame(frame_pixel, lenpath):
    normalizado = []
    amplified = []
    vet_var = []
    for i in range(lenpath):
        normalizado.append(frame_pixel[i])
        if(i>0 and frame_pixel[i] != frame_pixel[i-1]):
            vet_var.append([i, frame_pixel[i-1], frame_pixel[i+1]])
    for i in range(len(vet_var) - 1):
        casa = vet_var[i][0]
        casaNext = vet_var[i+1][0]
        if(casaNext - casa < 13):
            prevalue = vet_var[i][1]
            nextvalue = vet_var[i+1][2]
            if(prevalue == nextvalue):
                for x in range(casa, casaNext):
                    normalizado[x] = prevalue
    for i in range(lenpath):
        if(i == 0):
            amplified.append(normalizado[i])
        else:
            if(normalizado[i] == normalizado[i-1]):
                amplified.append(amplified[i - 1])
            else:
                if(normalizado[x] > normalizado[i - 1]):
                    delta = normalizado[i] - normalizado[i-1]
                    delta = 5 * delta
                    if((amplified[i-1] + delta)>255):
                        amplified.append(255)
                    else:
                        amplified.append(amplified[i-1] + delta)
                else:
                    delta = normalizado[i-1] - normalizado[i]
                    delta = 5 * delta
                    if((amplified[i-1] + delta)<0):
                        amplified.append(0)
                    else:
                        amplified.append(amplified[i-1] - delta)
    tempo = np.linspace(0, lenpath, num=lenpath, endpoint=True, dtype=int)
    plt.figure()
    #plt.xlabel
    #plt.ylabel
    #plt.title
    #plt.ylim
    plt.subplot(311)
    plt.plot(tempo, frame_pixel)
    plt.subplot(312)
    plt.plot(tempo, normalizado)
    plt.subplot(313)
    plt.plot(tempo, amplified)
    
def show_video(video):
    for i in range(180, 500):
        cv.imshow('figure 1' ,video[i])
        k = cv.waitKey(30)
        if(k == 33):
            break


def show_results(Original, Analise1, Analise2):
    for i in range(180, 500):
        cv.imshow('figure 1' ,Original[i])
        k = cv.waitKey(30)
        if(k == 33):
            break
    for i in range(180, 500):
        cv.imshow('figure 2' ,Analise1[i])
        k = cv.waitKey(30)
        if(k == 33):
            break    
    for i in range(180, 500):
        cv.imshow('figure 3' ,Analise2[i])
        k = cv.waitKey(30)
        if(k == 33):
            break

def amplify_video(Original, lenpath, altura, largura):
    Amplified = []
    for index in range(lenpath):
        Amplified.append(Original[index].copy())
    
    for index_height in range(altura):
        for index_width in range(largura):
            for index_frame in range(180, 500):
                value = int(Original[index_frame][index_height][index_width])
                prevalue = int(Original[index_frame - 1][index_height][index_width])
                if(value != prevalue):
                    dif = value - prevalue
                    dif = 5*dif
                    if(dif > 0):
                        value =  value + dif
                        if(value > 255):
                            value = 255
                        Amplified[index_frame][index_height][index_width] = value
                    else:
                        value = value + dif
                        if(value < 0):
                            value = 0
                        Amplified[index_frame][index_height][index_width] = value
                else:
                    Amplified[index_frame][index_height][index_width] = Amplified[index_frame - 1][index_height][index_width]
    return Amplified

        
    

path = './BaseVideos/max.avi'
lenpath = lenght_of_video(path)     #lenght do video
fpsVideo = get_fps(path)            #frames per second do video      
pixels = get_matrix_imgs(path)              #LISTA DE MATRIZ DE CADA FRAME
size_largura = size_x_matrix(pixels[0])     #pegar altura e largura do video
size_altura = size_y_matrix(pixels[0])     
p1 = get_frame(pixels, lenpath)

#result1 = amplify_video(pixels, lenpath, size_altura, size_largura)




#convert_to_video(pathExit, fpsVideo, video_exit, size_altura, size_largura, lenpath)

#encontrar formas de variaçoes pequenas serem ignoradas
#janela de frames anteriores
#verificar frames por segundo, talvez ignorar frames
#fixar camera para gravação
#variancia e desvio padrão
#np.var variancia, np.std desvio padrao
#destacar apenas onde variar (imagem nova com tudo 0)

#detecção de anomalias


