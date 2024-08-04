from ultralytics import YOLO
import cv2
import time

# Carregar o modelo YOLO
model = YOLO("ModeloReciclagem.pt")
camera = cv2.VideoCapture(0)
img_counter = 0

interval = 5  # Intervalo em segundos entre as capturas automáticas
last_capture_time = time.time()

while True:
    ret, frame = camera.read()
    if not ret:
        print("failed to grab frame")
        break
    
    cv2.imshow("Live Feed", frame)
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    current_time = time.time()
    if current_time - last_capture_time >= interval:
        # Tempo suficiente passou, capturar e processar a imagem
        img_path = "path/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_path, frame)
        
        # Realizar a detecção de objetos diretamente no frame
        results = model(frame)
        
        # Processar e exibir os resultados
        for result in results:
            boxes = result.boxes  # Objeto Boxes para os resultados de bbox
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Coordenadas do retângulo de bounding box
                conf = box.conf[0]  # Confiança da detecção
                cls = box.cls[0]  # Classe do objeto detectado
                label = f"{model.names[int(cls)]} {conf:.2f}"
                
                # Armazenar a classe detectada
                detected_class = model.names[int(cls)]
                object_detected = True
                
                # Definir a cor da caixa com base na classe detectada
                if detected_class == "papel":
                    color = (255, 0, 0)  # Azul
                elif detected_class == "metal":
                    color = (0, 255, 255)  # Amarelo
                elif detected_class == "plastico":
                    color = (0, 0, 255)  # Vermelho
                elif detected_class == "vidro":
                    color = (0, 255, 0)  # Verde
                else:
                    color = (255, 255, 255)  # Branco para classes desconhecidas
                
                # Desenhar a bounding box e o rótulo no frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if object_detected:
                    print("Objeto identificado:", detected_class)
                    detecting_object = True  # Sinalizar detecção em andamento

                    # Ações para as classes detectadas
                    if detected_class == "plastico":
                        print("Ação para plastico")
                    elif detected_class == "papel":
                        print("Ação para papel")
                    elif detected_class == "vidro":
                        print("Ação para vidro")
                    elif detected_class == "metal":
                        print("Ação para metal")
                    else:
                        print("Classe não reconhecida.")

                    object_detected = False  # Resetar flag de detecção

        # Mostrar o frame com as detecções em uma nova janela
        cv2.imshow("Detected Objects", frame)
        
        # Manter a janela aberta por 6 segundos
        cv2.waitKey(6000)
        cv2.destroyWindow("Detected Objects")
        
        img_counter += 1
        last_capture_time = current_time

camera.release()
cv2.destroyAllWindows()
