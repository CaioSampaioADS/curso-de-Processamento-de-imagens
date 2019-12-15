import cv2
'''
imagem = cv2.imread("limiar.png")
'''

'''
Pontos na Imagem

for i in range(0, imagem.shape[0], 7):
    for j in range(0, imagem.shape[1],7):
        if imagem[i][j][0] == 0:
            imagem[i, j] = (255,255,255)
'''

im

'''
desenhos unicos
imagem[30:50, 30:50] = (255, 0, 0)
'''



'''
escrevendo na imagem
fonte = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imagem,'OpenCV',(15,65), fonte,
2,(1,1,1),2,cv2.LINE_AA)
'''


'''
imagem = imagem[100:200, 100:200]
'''



'''
Limiar
for i in range(0, imagem.shape[0]):
    for h in range(0, imagem.shape[1]):
        if imagem[i][h][0] != 0:

            imagem[i, h] = (255, 255, 255)

'''



'''
REDIMENCIONAR
largura = imagem.shape[1]
altura = imagem.shape[0]
proporcao = float(altura/largura)
larguraNova = 320
alturaNova = int(larguraNova*proporcao)

imagem = cv2.resize(imagem, (larguraNova, alturaNova), interpolation=cv2.INTER_AREA)
'''


'''
Rotacionar
imagem = imagem[:, ::-1]
'''


'''
Mostrando canais de cores

imagemCores = cv2.imread("imagem.jpg")
(b,g,r) = cv2.split(imagemCores)



cv2.imshow("azul", r)
cv2.waitKey(0)
'''


'''
import matplotlib.pyplot as plt
resultadoAzul = []
resultadoVerde = []
resultadoVermelho = []
listaAzul = []
listaVerde = []
listaVermelho = []
imagem = cv2.imread("imagem.jpg")

#criando as listas para serem somadas
for i in range(0, imagem.shape[0]):
    for h in range(0, imagem.shape[1]):
        listaAzul.append(imagem[i][h][0])
        listaVerde.append(imagem[i][h][1])
        listaVermelho.append(imagem[i][h][2])

#Somando quantidade de valores do azul
listaSemAzul = sorted(set(listaAzul))

for i in range(0, len(listaSemAzul)):
    somatoria = 0
    for j in range(0, len(listaAzul)):
        if listaSemAzul[i] == listaAzul[j]:
            somatoria +=1
    resultadoAzul.append(somatoria)

#somando quantidade de valores verde
listaSemVerde = sorted(set(listaVerde))

for i in range(0, len(listaSemVerde)):
    somatoria = 0
    for j in range(0, len(listaVerde)):
        if listaSemVerde[i] == listaVerde[j]:
            somatoria +=1
    resultadoVerde.append(somatoria)

#somando Vermelho
listaSemVermelho = sorted(set(listaVermelho))

for i in range(0, len(listaSemVermelho)):
    somatoria = 0
    for j in range(0, len(listaVermelho)):
        if listaSemVermelho[i] == listaVermelho[j]:
            somatoria +=1
    resultadoVermelho.append(somatoria)





plt.plot(resultadoVermelho, color='red')
plt.plot(resultadoVerde, color='green')
plt.plot(resultadoAzul, color='blue')
plt.show()
'''

'''
REDIMENCIONAR
largura = imagem.shape[1]
altura = imagem.shape[0]
proporcao = float(altura/largura)
larguraNova = 320
alturaNova = int(larguraNova*proporcao)

imagem = cv2.resize(imagem, (larguraNova, alturaNova), interpolation=cv2.INTER_AREA)
'''


'''
import numpy as np
imagem = cv2.imread('imagem.jpg')
imagem = imagem[::1, ::1]
for i in range(0, imagem.shape[0], 30):
    for h in range(0, imagem.shape[1], 30):
        imagem[i:i+1,h:h+1] = (255, 255, 255)

suave = cv2.blur(imagem, (5, 5))
suave2 = cv2.blur(imagem, (11,11))

final = np.concatenate((imagem, suave, suave2), axis=1)

cv2.imshow("Imagem suavizada", final)
'''



'''
import numpy as np
imagem = cv2.imread('imagem.jpg')
imagem = imagem[::1, ::1]
for i in range(0, imagem.shape[0], 30):
    for h in range(0, imagem.shape[1], 30):
        imagem[i:i+3,h:h+3] = (255, 255, 255)

suave = cv2.medianBlur(imagem, 5)
suave2 = cv2.medianBlur(imagem, 11)

final = np.concatenate((imagem, suave, suave2), axis=1)

cv2.imshow("Imagem suavizada", final)
cv2.waitKey(0)
'''



'''
import numpy as np
import cv2
azulEscuro = np.array([100, 67, 0], dtype = "uint8")
azulClaro = np.array([255, 128, 50], dtype = "uint8")
#camera = cv2.VideoCapture('videoAzul.mp4')
camera = cv2.VideoCapture(0)
while True:
    (sucesso, frame) = camera.read()
    if not sucesso:
        break
    obj = cv2.inRange(frame, azulEscuro, azulClaro)

    (cnts, _) = cv2.findContours(obj.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(frame, [rect], -1, (0, 255, 255),2)
    cv2.imshow("Tracking", frame)
    cv2.imshow("Binary", obj)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break



camera.release()
cv2.destroyAllWindows()
'''



'''
import cv2
def redim(img, largura): #função para redimensionar uma imagem
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation = cv2.INTER_AREA)
    return img
#Cria o detector de faces baseado no XML
df = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#Abre um vídeo gravado em disco

camera = cv2.VideoCapture('video3.mp4')

#Também é possível abrir a próprio webcam
#do sistema para isso segue código abaixo
#camera = cv2.VideoCapture(0)
while True:
#read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = camera.read()
    if not sucesso: #final do vídeo
        break
#reduz tamanho do frame para acelerar processamento
    frame = redim(frame, 320)
#converte para tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#detecta as faces no frame
    faces = df.detectMultiScale(frame_pb, scaleFactor = 1.1, minNeighbors=3, minSize=(20,20), flags=cv2.CASCADE_SCALE_IMAGE)
    frame_temp = frame.copy()
    for (x, y, lar, alt) in faces:
        cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0, 255, 255), 2)
#Exibe um frame redimensionado (com perca de qualidade)
    cv2.imshow("Encontrando faces...", redim(frame_temp, 640))
#Espera que a tecla 's' seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break
#fecha streaming
camera.release()
cv2.destroyAllWindows()



https://www.youtube.com/watch?v=hPCTwxF0qf4
'''


'''


cv2.imshow("imagem", imagem)
cv2.waitKey(0)
'''

