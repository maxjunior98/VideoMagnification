import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#import statistics as sts

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

def show_video(video, lenapath):
    for i in range(lenpath):
        cv.imshow('figure 1' ,video[i])
        k = cv.waitKey(30)
        if(k == 33):
            break
        
def average_list(lst):
    return sum(lst)/len(lst)
       
def Accuracy(Original, Resultado):
    var = 0
    for i in range(len(Original)):
        if(Original[i] == Resultado[i]):
            var = var + 1
    var = var/len(Original)
    return var

def export_new_video(name, size_wid, size_hei, matrix_list, lenpath):
    out = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 30, (size_wid, size_hei), 0)
    for i in range(lenpath):
        out.write(matrix_list[i])
    out.release()


path = '../results/Apenas Regiao do Motor/ka1motor_magnification.avi'
lenpath = lenght_of_video(path)         #lenght do video
fpsVideo = get_fps(path)                #frames per second do video      
pixels = get_matrix_imgs(path)              #LISTA DE MATRIZ DE CADA FRAME
size_largura = size_x_matrix(pixels[0])     #pegar altura e largura do video
size_altura = size_y_matrix(pixels[0])

media = []
for i in range(20, lenpath-20):
    x = np.matrix(pixels[i])
    m = x.mean()
    media.append(m)

tempo = np.linspace(0, lenpath-40, num=lenpath-40, endpoint=True, dtype=int)
plt.figure()
plt.subplot(111)
plt.plot(tempo, media)

max_value = max(media)
max_value_index = media.index(max_value) + 20   #ENGINE START
avg = average_list(media)
DesPad = np.std(media)
Variance = np.var(media)

x = np.matrix(pixels[max_value_index])
x = x.astype(int)
xb = np.matrix(pixels[max_value_index - 1])
xb = xb.astype(int)

x = x - xb
max_h = int(np.argmax(x, axis=0).mean())

Max_Region = []
Media_Max = []
for i in range(lenpath):
    Max_Region.append(pixels[i][max_h-125:max_h+125,:])
    Media_Max.append(Max_Region[i].mean())
    
k = int(size_largura/5)
Region1 = []
MediaR1 = []
for i in range(lenpath):
    Region1.append(Max_Region[i][:,0:k-1])
    MediaR1.append(Region1[i].mean())
Region2 = []
MediaR2 = []
for i in range(lenpath):
    Region2.append(Max_Region[i][:,k:2*k-1])
    MediaR2.append(Region2[i].mean())
Region3 = []
MediaR3 = []
for i in range(lenpath):
    Region3.append(Max_Region[i][:,2*k:3*k-1])
    MediaR3.append(Region3[i].mean())
Region4 = []
MediaR4 = []
for i in range(lenpath):
    Region4.append(Max_Region[i][:,3*k:4*k-1])
    MediaR4.append(Region4[i].mean())
Region5 = []
MediaR5 = []
for i in range(lenpath):
    Region5.append(Max_Region[i][:,4*k:size_largura])
    MediaR5.append(Region5[i].mean())

tempo2 = np.linspace(0, lenpath, num=lenpath, endpoint=True, dtype=int)
plt.figure()
plt.subplot(211)
plt.plot(tempo2, Media_Max)
plt.subplot(212)
plt.plot(tempo2, MediaR1)

plt.figure()
plt.subplot(211)
plt.plot(tempo2, MediaR2)
plt.subplot(212)
plt.plot(tempo2, MediaR3)

plt.figure()
plt.subplot(211)
plt.plot(tempo2, MediaR4)
plt.subplot(212)
plt.plot(tempo2, MediaR5)

idx = 0
FreqR1 = []
FreqR2 = []
FreqR3 = []
FreqR4 = []
FreqR5 = []
k = 5
Data = []
while(idx < lenpath):
    if(idx+k >= lenpath):
        v = np.abs(np.fft.fft(MediaR1[idx:lenpath-1])).mean()
        FreqR1.append(v)
        v = np.abs(np.fft.fft(MediaR2[idx:lenpath-1])).mean()
        FreqR2.append(v)
        v = np.abs(np.fft.fft(MediaR3[idx:lenpath-1])).mean()
        FreqR3.append(v)
        v = np.abs(np.fft.fft(MediaR4[idx:lenpath-1])).mean()
        FreqR4.append(v)
        v = np.abs(np.fft.fft(MediaR5[idx:lenpath-1])).mean()
        FreqR5.append(v)
    else:    
        v = np.abs(np.fft.fft(MediaR1[idx:idx+k-1])).mean()
        FreqR1.append(v)
        v = np.abs(np.fft.fft(MediaR2[idx:idx+k-1])).mean()
        FreqR2.append(v)
        v = np.abs(np.fft.fft(MediaR3[idx:idx+k-1])).mean()
        FreqR3.append(v)
        v = np.abs(np.fft.fft(MediaR4[idx:idx+k-1])).mean()
        FreqR4.append(v)
        v = np.abs(np.fft.fft(MediaR5[idx:idx+k-1])).mean()
        FreqR5.append(v)
    if(idx < max_value_index and idx+k < max_value_index):
        Data.append(0)
    else:
        Data.append(1)
    idx = idx + k
        
FreqR1 = np.array(FreqR1)
FreqR2 = np.array(FreqR2)
FreqR3 = np.array(FreqR3)
FreqR4 = np.array(FreqR4)
FreqR5 = np.array(FreqR5)
Fator1 = FreqR1.mean() - 0.6*np.std(FreqR1)
Fator2 = FreqR2.mean() - 0.6*np.std(FreqR2)
Fator3 = FreqR3.mean() - 0.6*np.std(FreqR3)
Fator4 = FreqR4.mean() - 0.6*np.std(FreqR4)
Fator5 = FreqR5.mean() - 0.6*np.std(FreqR5)
        
Result = []
for j in range(len(FreqR1)):
    Votos = 0
    if(FreqR1[j]>=Fator1):
        Votos = Votos + 1
    if(FreqR2[j]>=Fator2):
        Votos = Votos + 1
    if(FreqR3[j]>=Fator3):
        Votos = Votos + 1
    if(FreqR4[j]>=Fator4):
        Votos = Votos + 1
    if(FreqR5[j]>=Fator5):
        Votos = Votos + 1
    Votos = Votos/5
    if(Votos > 0.5):
        Result.append(1)
    else:
        Result.append(0)

VideoClassificado = []
for i in range(lenpath):
    VideoClassificado.append(pixels[i])

j = 0
i = 0
while(j<lenpath):
    if(j + k >= lenpath):
        if(Result[i] == 0):
            for x in range(j,lenpath):
                VideoClassificado[x][0:50,:] = 0
        else:
            for x in range(j,lenpath):
                VideoClassificado[x][0:50,:] = 255
    else:
        if(Result[i] == 0):
            for x in range(j,j+k):
                VideoClassificado[x][0:50,:] = 0
        else:
            for x in range(j,j+k):
                VideoClassificado[x][0:50,:] = 255
    i = i + 1
    j = j + k

ResPrecisao = Accuracy(Data, Result)


# for j in range(max_value_index-10, max_value_index+11):
#     VideoClassificado[j][0:50,:] = 127
                




