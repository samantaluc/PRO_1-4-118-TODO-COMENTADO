import cv2

# Carregue o arquivo do classificador cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defina um objeto VideoCapture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture o vídeo quadro a quadro
    ret, frame = video_capture.read()

    if not ret:
        break

    # Converta para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte os rostos e os olhos
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Desenhe o retângulo ao redor de cada rosto e olhos
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Exiba o quadro resultante
    cv2.imshow('frame', frame)

    # Saia da tela ao pressionar a barra de espaço
    if cv2.waitKey(25) == 32:
        break

# Após o loop, libere o objeto capturado
video_capture.release()

# Destrua todas as janelas
cv2.destroyAllWindows()
