import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from copy import copy, deepcopy
import time

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

def get_pixel(matrix, lenpath, indexH, indexW):
    matrix_exit = []
    for i in range(lenpath):
       matrix_exit.append(int(matrix[i][indexH][indexW]))
    matrix_exit = np.array(matrix_exit) 
    return matrix_exit

def analise_pixel(frame_pixel, lenpath):
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
                    delta = 10 * delta
                    if((amplified[i-1] + delta)>255):
                        amplified.append(255)
                    else:
                        amplified.append(amplified[i-1] + delta)
                else:
                    delta = frame_pixel[i-1] - frame_pixel[i]
                    delta = 10 * delta
                    if((amplified[i-1] + delta)<0):
                        amplified.append(0)
                    else:
                        amplified.append(amplified[i-1] - delta)
    tempo = np.linspace(0, lenpath, num=lenpath, endpoint=True, dtype=int)
    maxlabel = max(amplified) + 2
    minlabel = min(amplified) - 2
    plt.figure()
    plt.ylim(minlabel, maxlabel)
    a = plt.subplot(211)
    a.set_ylim([minlabel, maxlabel])
    a.set_ylabel('Amplitude')
    a.set_xlabel('Frames')
    a.title.set_text('a)')
    plt.plot(tempo, frame_pixel)
    b = plt.subplot(212)
    b.set_ylabel('Amplitude')
    b.set_xlabel('Frames')
    b.title.set_text('b)')
    b.set_ylim([minlabel, maxlabel])
    plt.plot(tempo, amplified)
    return amplified
    
    
def analise_tecnica(frame_pixel, lenpath):
    amplified = []
    for i in range(lenpath):
        if(i == 0):
            amplified.append(frame_pixel[i])
        else:
            delta = frame_pixel[i] - frame_pixel[i - 1]
            if(delta == 0):
                amplified.append(frame_pixel[i-1])
            elif(np.abs(delta) < 4):
                amplified.append(frame_pixel[i])
            else:
                delta = 10*delta
                value = frame_pixel[i] + delta
                if(value > 255):
                    amplified.append(255)
                elif(value<0):
                    amplified.append(0)
                else:
                    amplified.append(value)
    tempo = np.linspace(0, lenpath, num=lenpath, endpoint=True, dtype=int)
    maxlabel = max(amplified) + 2
    minlabel = min(amplified) - 2
    plt.figure()
    plt.ylim(minlabel, maxlabel)
    a = plt.subplot(211)
    a.set_ylabel('Amplitude')
    a.set_xlabel('Frames')
    a.title.set_text('a)')
    a.set_ylim([minlabel, maxlabel])
    plt.plot(tempo, frame_pixel)
    b = plt.subplot(212)
    b.set_ylabel('Amplitude')
    b.set_xlabel('Frames')
    b.title.set_text('b)')
    b.set_ylim([minlabel, maxlabel])
    plt.plot(tempo, amplified)
    return amplified
    

def Analise_Original(pixel, lenpath):
    tempo = np.linspace(0, lenpath, num=lenpath, endpoint=True, dtype=int)
    plt.figure()
    plt.title('Pixel x Frame')
    plt.ylabel('Amplitude')
    plt.xlabel('Frame')
    plt.plot(tempo, pixel)


def show_video(video, lenapath):
    for i in range(lenpath):
        cv.imshow('figure 1' ,video[i])
        k = cv.waitKey(30)
        if(k == 33):
            break

def video_magnification(Original, lenpath, altura_inicial, altura_final, largura):
    Amplified = []
    for index in range(lenpath):
        Amplified.append(Original[index].copy())
    
    for index_height in range(altura_inicial, altura_final):
        for index_width in range(largura):
            for index_frame in range(lenpath):
                value = int(Original[index_frame][index_height][index_width])
                prevalue = int(Original[index_frame - 1][index_height][index_width])
                if(value != prevalue):
                    dif = value - prevalue
                    if(np.abs(dif)> 4):
                        dif = 10*dif
                        value = value + dif
                        if(value > 255):
                            value = 255
                        elif(value < 0):
                            value = 0
                        Amplified[index_frame][index_height][index_width] = value
                    else:
                        Amplified[index_frame][index_height][index_width] = Original[index_frame][index_height][index_width]
                else:
                    Amplified[index_frame][index_height][index_width] = Original[index_frame - 1][index_height][index_width]
    return Amplified


def wrapper_magnification(Original, amp, largura, length, startAt, altura, i):
    Amplified = np.array(deepcopy(Original))

    for index_height in range(startAt, altura):
        for index_width in range(largura):
            for index_frame in range(length):
                value = int(Original[index_frame][index_height][index_width])
                prevalue = int(Original[index_frame - 1][index_height][index_width])
                if(value != prevalue):
                    dif = value - prevalue
                    if(np.abs(dif)> 4):
                        dif = 10*dif
                        value = value + dif
                        if(value > 255):
                            value = 255
                        elif(value < 0):
                            value = 0
                        Amplified[index_frame][index_height][index_width] = value
                    else:
                        Amplified[index_frame][index_height][index_width] = Original[index_frame][index_height][index_width]
                else:
                    Amplified[index_frame][index_height][index_width] = Original[index_frame - 1][index_height][index_width]
    
    amp[i] = Amplified[ : , startAt:altura, : ]

def multi_amplify(Original, length, altura_inicial, altura_final , largura, processes = 4):
    Amplification = []
    for index in range(length):
        Amplification.append(Original[index].copy())
    
    manager1 = multiprocessing.Manager()
    ampList = manager1.list()

    jobs = []
    altura = int((altura_final - altura_inicial) / processes)
    for i in range(processes):
        ampList.append([])
        p = multiprocessing.Process(target=wrapper_magnification, args=(Original, ampList, largura, length, 
                                    altura_inicial + i*altura, altura_inicial + (i+1)*altura, i))
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()

    Amplification = np.array(Amplification)
    for  i in range(processes):
        Amplification[:,altura_inicial + i*altura : altura_inicial + (i+1)*altura,:] = ampList[i]

    return Amplification
        
def export_new_video(name, size_wid, size_hei, matrix_list, lenpath):
    out = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 30, (size_wid, size_hei), 0)
    for i in range(lenpath):
        out.write(matrix_list[i])
    out.release()
    
def average_list(lst):
    return sum(lst)/len(lst)
    

path = './BaseVideos/renegade2020.mp4'
lenpath = lenght_of_video(path)             #lenght do video
fpsVideo = get_fps(path)                    #frames per second do video      
pixels = get_matrix_imgs(path)              #LISTA DE MATRIZ DE CADA FRAME
size_largura = size_x_matrix(pixels[0])     #pegar altura e largura do video
size_altura = size_y_matrix(pixels[0])
p1 = get_pixel(pixels, lenpath, 342, 440)
p2 = get_pixel(pixels, lenpath, 233, 250)

media = []
for i in range(20,lenpath-20):
    x = np.matrix(pixels[i])
    m = x.mean()
    media.append(m)
tempo = np.linspace(0, lenpath-40, num=lenpath-40, endpoint=True, dtype=int)
plt.figure()
plt.title('Média x Frame')
plt.ylabel('Média')
plt.xlabel('Frame')
plt.plot(tempo, media)


max_value = max(media)
max_value_index = media.index(max_value) + 20

x = np.matrix(pixels[max_value_index])
x = x.astype(int)
xb = np.matrix(pixels[max_value_index - 1])
xb = xb.astype(int)
x = x - xb
max_altura_motor = max(np.squeeze(np.asarray(np.argmax(x, axis=0))))
min_altura_motor = min(np.squeeze(np.asarray(np.argmax(x, axis=0))))
altura_motor = int((max_altura_motor - min_altura_motor)/2)

Motor_Region = []
for i in range(lenpath):
    Motor_Region.append(pixels[i][altura_motor-150:altura_motor+150,:])

#print(altura_motor +150 - altura_motor -150)
#result = video_magnification(pixels, lenpath, altura_motor - 150, altura_motor + 150, size_largura)
t0 = time.time()
result = video_magnification(pixels, lenpath, altura_motor - 150, altura_motor + 150, size_largura)
t1 = time.time()
total1 = t1 - t0

t0 = time.time()
result = multi_amplify(pixels, lenpath, altura_motor - 150, altura_motor + 150, size_largura, 2)
t1 = time.time()
total2 = t1 - t0

t0 = time.time()
result = multi_amplify(pixels, lenpath, altura_motor - 150, altura_motor + 150, size_largura, 3)
t1 = time.time()
total3 = t1 - t0

t0 = time.time()
result = multi_amplify(pixels, lenpath, altura_motor - 150, altura_motor + 150, size_largura, 4)
t1 = time.time()
total4 = t1 - t0

print(total1)
print(total2)
print(total3)
print(total4)

# smo = smoothing(pixels, lenpath)


#encontrar formas de variaçoes pequenas serem ignoradas
#janela de frames anteriores
#verificar frames por segundo, talvez ignorar frames
#fixar camera para gravação
#variancia e desvio padrão
#np.var variancia, np.std desvio padrao
#destacar apenas onde variar (imagem nova com tudo 0)


#detecção de anomalias


