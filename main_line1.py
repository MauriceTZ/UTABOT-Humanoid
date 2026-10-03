import cv2
import numpy as np
import argparse
import time
from picamera2 import Picamera2

def inicializar_camara(ancho=320, alto=240):
    picam = Picamera2()
    
    # main = Lo que recibe OpenCV (pequeño y rápido de procesar)
    # raw = Truco para obligar al sensor físico a usar su máxima apertura de lente (Full FOV)
    config = picam.create_video_configuration(
        main={"size": (ancho, alto), "format": "BGR888"},
        raw={"size": (1296, 972)} 
    )
    
    picam.configure(config)
    picam.start()
    
    print("Calentando sensor CSI (OV5647) en modo Full-FOV...")
    time.sleep(2.0)
    
    return picam

def procesar_pista(modo_debug=False):
    try:
        picam = inicializar_camara()
    except Exception as e:
        print(f"Error crítico al inicializar la cámara: {e}")
        return

    if modo_debug:
        print("MODO DEBUG: Visualización en tiempo real activada. (Presiona 'q' para salir)")
    else:
        print("MODO PRODUCCIÓN: Ejecución silenciosa para motores.")

    # --- MEMORIA DEL ALGORITMO ---
    # Valor estimado inicial (en píxeles) del ancho de la pista para una resolución de 320x240
    ancho_pista_memoria = 200 

    try:
        while True:
            frame = picam.capture_array()
            alto, ancho = frame.shape[:2]

            recorte_top = int(alto * 0.25)
            roi = frame[recorte_top:alto, :]
            roi_alto = roi.shape[0]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mascara = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            meta_detectada = False
            error_trayectoria = None

            # 3. Detectar META
            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contornos:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if aspect_ratio > 4.0 and w > (ancho * 0.5) and area > 500:
                    meta_detectada = True
                    if modo_debug:
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    break 

            # 4. MUESTREO DE FILA (Scanline)
            linea_muestreo_y = int(roi_alto * 0.6)
            fila_pixeles = mascara[linea_muestreo_y, :]
            indices_blancos = np.where(fila_pixeles == 255)[0]

            if len(indices_blancos) > 0:
                diferencias = np.diff(indices_blancos)
                cortes = np.where(diferencias > 25)[0] 
                
                centro_robot_x = ancho // 2

                if len(cortes) > 0:
                    # ==========================================
                    # CASO A: Vio AMBAS líneas (Izquierda y Derecha)
                    # ==========================================
                    grupo_izq = indices_blancos[:cortes[0]+1]
                    grupo_der = indices_blancos[cortes[-1]+1:]
                    
                    linea_izq_x = int(np.mean(grupo_izq))
                    linea_der_x = int(np.mean(grupo_der))
                    
                    # Actualizar la memoria con la distancia real leída en este fotograma
                    ancho_pista_memoria = linea_der_x - linea_izq_x
                    
                    centro_ideal_x = (linea_izq_x + linea_der_x) // 2
                    error_trayectoria = centro_ideal_x - centro_robot_x
                    
                    if modo_debug:
                        cv2.circle(roi, (linea_izq_x, linea_muestreo_y), 4, (255, 0, 0), -1)
                        cv2.circle(roi, (linea_der_x, linea_muestreo_y), 4, (0, 255, 0), -1)
                        cv2.circle(roi, (centro_ideal_x, linea_muestreo_y), 4, (0, 255, 255), -1)
                        
                else:
                    # ==========================================
                    # CASO B: Vio UNA SOLA línea 
                    # ==========================================
                    linea_unica_x = int(np.mean(indices_blancos))
                    
                    if linea_unica_x < centro_robot_x:
                        # Asume que es la línea IZQUIERDA. Proyecta el centro hacia la derecha.
                        centro_ideal_x = linea_unica_x + (ancho_pista_memoria // 2)
                    else:
                        # Asume que es la línea DERECHA. Proyecta el centro hacia la izquierda.
                        centro_ideal_x = linea_unica_x - (ancho_pista_memoria // 2)
                        
                    error_trayectoria = centro_ideal_x - centro_robot_x
                        
                    if modo_debug:
                        cv2.circle(roi, (linea_unica_x, linea_muestreo_y), 5, (0, 165, 255), -1) # Naranja: Línea única
                        cv2.circle(roi, (centro_ideal_x, linea_muestreo_y), 4, (0, 255, 255), -1) # Amarillo: Centro proyectado virtualmente

            # 5. Volcado de Debug Visual en Tiempo Real
            if modo_debug:
                msg_error = f"{error_trayectoria}px" if error_trayectoria is not None else "PISTA PERDIDA"
                cv2.putText(roi, f"Error: {msg_error}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.line(roi, (0, linea_muestreo_y), (ancho, linea_muestreo_y), (255, 0, 255), 1)
                cv2.line(roi, (ancho//2, 0), (ancho//2, roi_alto), (255, 255, 0), 1)
                
                cv2.imshow("Mascara del Algoritmo (Binarizada)", mascara)
                cv2.imshow("Vision del Robot", roi)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Saliendo del modo debug...")
                    break
                
    except KeyboardInterrupt:
        print("\nEjecución detenida manualmente por el usuario en terminal.")
    finally:
        picam.stop()
        picam.close()
        if modo_debug:
            cv2.destroyAllWindows()
        print("Recursos liberados correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    procesar_pista(modo_debug=args.debug)