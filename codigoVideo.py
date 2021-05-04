import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from copy import copy, deepcopy

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
            frame = frame [:,:,:]                           #PEGANDO PARTE DO VIDEO
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    #CONVERTE PARA GRAY
            matrix.append(gray)
        except:
            break
    cap.release()
    return matrix

def get_frame(matrix, lenpath, indexH, indexW):
    matrix_exit = []
    for i in range(lenpath):
       matrix_exit.append(int(matrix[i][indexH][indexW]))
    matrix_exit = np.array(matrix_exit) 
    return matrix_exit

def get_analise_frame(frame_pixel, lenpath):
    amplified = []
    for i in range(lenpath):
        if(i == 0):
            amplified.append(frame_pixel[i])
        else:
            if(frame_pixel[i] == frame_pixel[i-1]):
                amplified.append(amplified[i - 1])
            else:
                if(frame_pixel[i] > frame_pixel[i - 1]):
                    delta = frame_pixel[i] - frame_pixel[i-1]
                    delta = 5 * delta
                    if((amplified[i-1] + delta)>255):
                        amplified.append(255)
                    else:
                        amplified.append(amplified[i-1] + delta)
                else:
                    delta = frame_pixel[i-1] - frame_pixel[i]
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
    plt.subplot(211)
    plt.plot(tempo, frame_pixel)
    plt.subplot(212)
    plt.plot(tempo, amplified)
    
def show_video(video, lenapath):
    for i in range(lenpath):
        cv.imshow('figure 1' ,video[i])
        k = cv.waitKey(30)
        if(k == 33):
            break

def wrapper_simple(Original, amp, largura, length, startAt, altura, i):
    Amplified = np.array(deepcopy(Original))
    for index_height in range(startAt, altura):
        for index_width in range(largura):
            for index_frame in range(length):
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
    amp[i] = Amplified[ : , startAt:altura, : ]

def wrapper_lowF(Original, amp, largura, length, startAt, altura, i):
    Amplified = np.array(deepcopy(Original))
    for index_height in range(startAt, altura):
        for index_width in range(largura):
            for index_frame in range(length):
                value = int(Original[index_frame][index_height][index_width])
                prevalue = int(Original[index_frame - 1][index_height][index_width])
                if(value != prevalue):
                    dif = value - prevalue
                    if(dif < 4 and dif > -4):
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
    amp[i] = Amplified[ : , startAt:altura, : ]

def wrapper_highF(Original, amp, largura, length, startAt, altura, i):
    Amplified = np.array(deepcopy(Original))
    for index_height in range(startAt, altura):
        for index_width in range(largura):
            for index_frame in range(length):
                value = int(Original[index_frame][index_height][index_width])
                prevalue = int(Original[index_frame - 1][index_height][index_width])
                if(value != prevalue):
                    dif = value - prevalue
                    if(dif > 4 or dif < -4):
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
    amp[i] = Amplified[ : , startAt:altura, : ]

def multi_amplify(wrapper, Original, length, altura, largura):
    Amplification = []
    for index in range(length):
        Amplification.append(Original[index].copy())
    
    manager1 = multiprocessing.Manager()
    ampList = manager1.list()

    jobs = []
    altura = int(altura / 4)
    for i in range(4):
        ampList.append([])
        p = multiprocessing.Process(target=wrapper, args=(Original, ampList, largura, length, i*altura, (i+1)*altura, i))
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()

    Amplification = np.array(Amplification)
    Amplification[:,0:altura,:] = ampList[0]
    Amplification[:,altura:2*altura,:] = ampList[1]
    Amplification[:,2*altura:3*altura,:] = ampList[2]
    Amplification[:,3*altura:4*altura,:] = ampList[3]

    return Amplification

def multi_amplify_video_highF(Original, length, altura, largura):
    return multi_amplify(wrapper_highF, Original, length, altura, largura)

    
def multi_amplify_video_lowF(Original, length, altura, largura):
    return multi_amplify(wrapper_lowF, Original, length, altura, largura)

    
def multi_simple_amplify(Original, length, altura, largura):
    return multi_amplify(wrapper_simple, Original, length, altura, largura)

def amplify_video_highF(Original, lenpath, altura, largura):
    Amplified = []
    for index in range(lenpath):
        Amplified.append(Original[index].copy())
    
    for index_height in range(altura):
        for index_width in range(largura):
            for index_frame in range(lenpath):
                value = int(Original[index_frame][index_height][index_width])
                prevalue = int(Original[index_frame - 1][index_height][index_width])
                if(value != prevalue):
                    dif = value - prevalue
                    if(dif > 4 or dif < -4):
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


def amplify_video_lowF(Original, lenpath, altura, largura):
    Amplified = []
    for index in range(lenpath):
        Amplified.append(Original[index].copy())
    
    for index_height in range(altura):
        for index_width in range(largura):
            for index_frame in range(lenpath):
                value = int(Original[index_frame][index_height][index_width])
                prevalue = int(Original[index_frame - 1][index_height][index_width])
                if(value != prevalue):
                    dif = value - prevalue
                    if(dif < 4 and dif > -4):
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

def simple_amplify(Original, lenpath, altura, largura):
    Amplified = []
    for index in range(lenpath):
        Amplified.append(Original[index].copy())
        
    for index_height in range(altura):
        for index_width in range(largura):
            for index_frame in range(lenpath):
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


def change_direction(video, lenpath, col, line):
    newShape = []
    for i in range(lenpath):
        tryReshape = np.array([])
        tryReshape = video[i]
        tryReshape = tryReshape.transpose()
        reOrder = np.array([])
        reOrder = tryReshape.copy()
        for c in range(line):
            l1 = col - 1
            l2 = 0
            while(l1>=0):
                reOrder[l2][c] = tryReshape[l1][c]
                l1 = l1 - 1
                l2 = l2 + 1
        newShape.append(reOrder)
    return newShape
        
def export_new_video(name, size_wid, size_hei, matrix_list, lenpath):
    out = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 30, (size_wid, size_hei), 0)
    for i in range(lenpath):
        out.write(matrix_list[i])
    out.release()

path = './BaseVideos/renegade2020.mp4'
lenpath = lenght_of_video(path)     #lenght do video
fpsVideo = get_fps(path)            #frames per second do video      
pixels = get_matrix_imgs(path)              #LISTA DE MATRIZ DE CADA FRAME
size_largura = size_x_matrix(pixels[0])     #pegar altura e largura do video
size_altura = size_y_matrix(pixels[0])
p1 = get_frame(pixels, lenpath, 20, 20)
original = get_matrix_imgs(path)
amp = multi_amplify_video_highF(original,lenpath,size_altura, size_largura)
amplifiedLow = amplify_video_lowF(original,lenpath,size_altura, size_largura)
show_video(original, lenpath)
show_video(amp, lenpath)
cv.destroyAllWindows()
show_video(amplifiedLow, lenpath)