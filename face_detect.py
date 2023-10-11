# Importa a biblioteca OpenCV para processamento de imagens
import cv2

# Carrega a imagem "boy.jpg" para a variável 'img'
img = cv2.imread("boy.jpg")

# Converte a imagem colorida em escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Carrega o classificador em cascata para detecção de faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detecta faces na imagem em escala de cinza
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Imprime o número de faces detectadas
print(len(faces))

# Para cada face detectada, desenhe um retângulo ao redor dela
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Corte a imagem para salvar a imagem do rosto.
    roi_color = img[y:y+h, x:x+w]
    cv2.imwrite("face.jpg", roi_color)

# Exibe a imagem com retângulos ao redor das faces
cv2.imshow('img', img)

# Aguarda até que uma tecla seja pressionada para fechar a janela
cv2.waitKey(0)
