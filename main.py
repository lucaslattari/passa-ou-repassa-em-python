import numpy as np
import cv2
from playsound import playsound
import face_recognition

#mostra imagens intermediárias para teste
def mostra_imagem(img):
    cv2.imshow('janela de teste', img)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
    exit(0)

def inserir_torta_na_cara(frame, torta, face_locations):
    frameH, frameW, _ = frame.shape
    tortaH, tortaW, _ = torta.shape

    #sem face, sem torta
    if not face_locations:
        return

    #checa limites da tela, para que a torta não "saia" da janela
    offsetH, offsetW = face_locations[0][0] - int(frameH * 0.13), face_locations[0][1] - int(frameW * 0.35)
    if offsetW < 0:
        offsetW = 0
    if offsetH < 0:
        offsetH = 0
    if tortaW + offsetW > frameW:
        offsetW = frameW - tortaW - 10
    if tortaH + offsetH > frameH:
        offsetH = frameH - tortaH - 10

    crop = frame[offsetH:tortaH + offsetH, offsetW:tortaW + offsetW]
    #mostra_imagem(crop)

    torta_cinza = cv2.cvtColor(torta, cv2.COLOR_BGR2GRAY)
    _, torta_mask = cv2.threshold(torta_cinza, 100, 255, cv2.THRESH_BINARY)
    torta_mask_inv = cv2.bitwise_not(torta_mask)
    #mostra_imagem(torta_mask_inv)

    fundo = cv2.bitwise_and(crop, crop, mask = torta_mask_inv)
    #print(crop.shape, torta_mask.shape)
    #mostra_imagem(fundo)

    frente = cv2.bitwise_and(torta, torta, mask = torta_mask)
    #mostra_imagem(frente)

    imgJunta = cv2.add(frente, fundo)
    #mostra_imagem(imgJunta)

    frame[offsetH:tortaH + offsetH, offsetW:tortaW + offsetW] = imgJunta
    #mostra_imagem(frame)

    return frame

#main
if __name__ == '__main__':
    #carrega imagem da torta
    torta = cv2.imread('torta.png')

    #redimensiona torta de acordo com webcam
    dsize = (int(torta.shape[0] * 0.20), int(torta.shape[1] * 0.20))
    torta = cv2.resize(torta, dsize, interpolation = cv2.INTER_AREA)

    #webcam
    cap = cv2.VideoCapture(0)

    #pergunta
    resposta = input("Qual o melhor canal de Python?")
    if resposta == "Guanabara":
        print("Certa a resposta!")
    else:
        print("Errooooou!")

    it = 0
    while(True):
        # captura frame a frame
        ret, frame = cap.read()

        #a cada cinco quadros
        if it % 5 == 0:
            #pega as coordenadas da face
            face_locations = face_recognition.face_locations(frame)

        #o próprio nome já diz hehe
        inserir_torta_na_cara(frame, torta, face_locations)

        #nos primeiros 10 quadros faz o barulho de torta
        if it == 10:
            playsound('torta.mp3')
            flag_torta = 1

        #mostra o frame
        cv2.imshow('frame', frame)

        #se apertar q, sai
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        it += 1

    #finaliza
    cap.release()
    cv2.destroyAllWindows()
