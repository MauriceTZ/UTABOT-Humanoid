import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from ikpy.chain import Chain
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS, GroupSyncWrite
from marcha import ParamsMarcha, GeneradorMarcha


# ==========================================
# 1. MAPA DE MEMORIA DYNAMIXEL (XL430-W250-T)
# ==========================================
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_P_GAIN = 84
ADDR_PROFILE_ACCEL = 108
ADDR_PROFILE_VEL = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# ==========================================
# 2. CONFIGURACIÓN DE HARDWARE
# ==========================================
SERVO_IDS_IZQ = [6, 7, 8, 9, 10, 11]
SERVO_IDS_DER = [5, 4, 3, 2, 1, 0]
PUERTO = "/dev/ttyUSB0"
BAUD_RATE = 1000000

PROFILE_ACCEL = 0        # 0 = sin límite (la trayectoria ya es suave)
PROFILE_VEL = 0          # 0 = sin límite; se limita por software con VEL_MAX_SOFT
VEL_MAX_SOFT = 6.0       # rad/s, tope de seguridad por servo
T_AGACHARSE = 2.0        # s para pasar de pierna recta a la postura de marcha
CICLOS_RAMPA = 2.0       # ciclos para llegar a la amplitud completa


def sin_ramp(t, a=1):
    s = math.sin(t)
    return math.copysign(abs(s)**a, s)


def rad_a_dynamixel(angulo_rad, centro_fisico, invertir=False):
    direccion = -1 if invertir else 1
    pasos = int(centro_fisico + (direccion *
                angulo_rad * 4096 / (2 * math.pi)))
    return max(-1048575, min(1048575, pasos))


def dxl_to_signed(val):
    if val > 0x7fffffff:
        val -= 4294967296
    return val


def matriz_rotacion_rpy(roll, pitch, yaw):
    """
    Genera una matriz de rotación 3x3 (SO3) a partir de ángulos en radianes.
    Aplicable al pie: Roll (inclinación lateral), Pitch (punta/talón), Yaw (giro direccional).
    """
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Multiplicación de matrices en orden ZYX
    return Rz @ Ry @ Rx


# ==========================================
# 3. INICIALIZACIÓN DE CINEMÁTICA (ikpy)
# ==========================================
mascara_activos = [False, True, True, True, True, True, True]

nombres_izq = [
    "Cuerpo1:12_link_joint", "Cuerpo1:8_link_joint", "Cuerpo1:6_link_joint",
    "Cuerpo1:4_link_joint", "Cuerpo1:3_link_joint", "Cuerpo1:1_link_joint"
]
pierna_izquierda = Chain.from_urdf_file(
    "Humanoid/urdf/Humanoid.urdf",
    base_elements=["base_link"] + [item for par in zip(
        nombres_izq, [n.replace("_joint", "") for n in nombres_izq]) for item in par],
    active_links_mask=mascara_activos
)

nombres_der = [
    "Cuerpo1:25_link_joint", "Cuerpo1:21_link_joint", "Cuerpo1:19_link_joint",
    "Cuerpo1:17_link_joint", "Cuerpo1:16_link_joint", "Cuerpo1:14_link_joint"
]
pierna_derecha = Chain.from_urdf_file(
    "Humanoid/urdf/Humanoid.urdf",
    base_elements=["base_link"] + [item for par in zip(
        nombres_der, [n.replace("_joint", "") for n in nombres_der]) for item in par],
    active_links_mask=mascara_activos
)

# Límites articulares (rad) en orden: cad_roll, cad_pitch, rodilla, tob_pitch, tob_yaw, tob_roll.
# La rodilla queda SIEMPRE doblada hacia adelante: sin esto ikpy puede saltar a la solución
# "rodilla hacia atrás" (pasa cuando el pie queda ~3 cm adelante de la cadera) y el robot da un tirón.
LIMITES_ARTIC = [(-0.6, 0.6), (-1.6, 1.6), (-2.4, -0.05), (-1.6, 1.6), (-0.8, 0.8), (-0.6, 0.6)]
for cadena in (pierna_izquierda, pierna_derecha):
    for i, lim in enumerate(LIMITES_ARTIC):
        cadena.links[i + 1].bounds = lim

# Semilla de la IK con la rodilla ya doblada (evita arrancar desde la singularidad de pierna recta)
SEMILLA_IK = [0.0, 0.0, 0.4, -0.8, 0.4, 0.0, 0.0]
angulos_previos_izq = list(SEMILLA_IK)
angulos_previos_der = list(SEMILLA_IK)

matriz_fk_izq = pierna_izquierda.forward_kinematics([0.0] * 7)
matriz_fk_der = pierna_derecha.forward_kinematics([0.0] * 7)
orientacion_plana_izq = matriz_fk_izq[:3, :3]
orientacion_plana_der = matriz_fk_der[:3, :3]
# Posición del punto IK del pie con la pierna recta (sale del URDF, no se hardcodea)
neutro_izq = matriz_fk_izq[:3, 3]
neutro_der = matriz_fk_der[:3, 3]

# ==========================================
# 4. INICIALIZACIÓN DE MUJOCO
# ==========================================
model = mujoco.MjModel.from_xml_path("Humanoid/urdf/Humanoid.urdf")
data = mujoco.MjData(model)

indices_qpos_izq = [model.jnt_qposadr[mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, nombre)] for nombre in nombres_izq]
indices_qpos_der = [model.jnt_qposadr[mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, nombre)] for nombre in nombres_der]

# ==========================================
# 5. FUNCIÓN PRINCIPAL DE MARCHA
# ==========================================


def iniciar_robot(visualizar=False, params=None):
    viewer = None

    # Solo los servos detectados entrarán a estos diccionarios
    centros_fisicos_izq = {}
    centros_fisicos_der = {}

    config_giro_izq = {6: True, 7: False,
                       8: False, 9: False, 10: False, 11: False}
    config_giro_der = {0: False, 1: False,
                       2: False, 3: False, 4: False, 5: True}

    portHandler = PortHandler(PUERTO)
    packetHandler = PacketHandler(2.0)

    if not portHandler.openPort():
        print(f"Error crítico: No se pudo abrir {PUERTO}.")
        return
    if not portHandler.setBaudRate(BAUD_RATE):
        print(f"Error crítico: No se pudo establecer Baudrate a {BAUD_RATE}.")
        return

    print("Sostén las piezas conectadas en la posición CERO. Capturando offsets en 3 segundos...")
    time.sleep(3)
    print("Iniciando escaneo de hardware...")

    todos_los_servos = SERVO_IDS_IZQ + SERVO_IDS_DER

    for servo_id in todos_los_servos:
        # 1. Hacemos "Ping" leyendo la posición primero. Si falla, el servo no está conectado.
        offset, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, servo_id, ADDR_PRESENT_POSITION)

        if dxl_comm_result == COMM_SUCCESS:
            # 2. Si responde, procedemos a configurarlo con seguridad
            packetHandler.write1ByteTxRx(
                portHandler, servo_id, ADDR_TORQUE_ENABLE, 0)
            packetHandler.write1ByteTxRx(
                portHandler, servo_id, ADDR_OPERATING_MODE, 4)

            packetHandler.write2ByteTxRx(
                portHandler, servo_id, ADDR_POSITION_P_GAIN, 800)
            packetHandler.write2ByteTxRx(
                portHandler, servo_id, ADDR_POSITION_D_GAIN, 0)
            # La trayectoria ya llega suave a 60 Hz: un PROFILE_ACCEL bajo (50 = ~19 rad/s²)
            # la filtra y atrasa (la marcha pide 50-80 rad/s²). Se deja amplio y se limita
            # la velocidad por software más abajo.
            packetHandler.write4ByteTxRx(
                portHandler, servo_id, ADDR_PROFILE_ACCEL, PROFILE_ACCEL)
            packetHandler.write4ByteTxRx(
                portHandler, servo_id, ADDR_PROFILE_VEL, PROFILE_VEL)

            offset = dxl_to_signed(offset)

            # Registrarlo en la lista de activos
            if servo_id in SERVO_IDS_IZQ:
                centros_fisicos_izq[servo_id] = offset
            else:
                centros_fisicos_der[servo_id] = offset

            packetHandler.write4ByteTxRx(
                portHandler, servo_id, ADDR_GOAL_POSITION, offset & 0xFFFFFFFF)
            packetHandler.write1ByteTxRx(
                portHandler, servo_id, ADDR_TORQUE_ENABLE, 1)
            print(
                f"✅ Servo {servo_id} detectado y calibrado | Offset: {offset}")
        else:
            # Si no responde, simplemente lo saltamos
            print(
                f"⚠️ Advertencia: Servo {servo_id} ausente. Se omitirá en esta sesión.")

    if not centros_fisicos_izq and not centros_fisicos_der:
        print("Falla crítica: Ningún servo detectado en el bus. Cerrando programa.")
        portHandler.closePort()
        return

    # Parámetros de marcha (ver marcha.py). Ajustar con validar_marcha.py antes de probar en el robot
    params = params or ParamsMarcha()
    generador = GeneradorMarcha(params, neutro_izq, neutro_der)
    print("Marcha:", generador.resumen())

    if visualizar:
        print("Lanzando entorno gráfico MuJoCo...")
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 60.0
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = -12.5
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 45.0
    else:
        print("Ejecutando en MODO PRODUCCIÓN (Sin GUI). Presiona Ctrl+C para detener.")

    global angulos_previos_izq, angulos_previos_der

    def resolver_ik(target_izq, yaw_izq, target_der, yaw_der):
        global angulos_previos_izq, angulos_previos_der
        R_izq = matriz_rotacion_rpy(0.0, 0.0, yaw_izq) @ orientacion_plana_izq
        R_der = matriz_rotacion_rpy(0.0, 0.0, yaw_der) @ orientacion_plana_der
        sol_izq = pierna_izquierda.inverse_kinematics(
            target_position=target_izq, target_orientation=R_izq,
            orientation_mode="all", initial_position=angulos_previos_izq)
        sol_der = pierna_derecha.inverse_kinematics(
            target_position=target_der, target_orientation=R_der,
            orientation_mode="all", initial_position=angulos_previos_der)
        angulos_previos_izq, angulos_previos_der = sol_izq, sol_der
        return sol_izq, sol_der

    # Postura de pie (fase 0 con intensidad 0) y su solución IK
    pie_izq0, _, pie_der0, _ = generador.objetivos(0.0, intensidad=0.0)
    postura_izq, postura_der = resolver_ik(pie_izq0, 0.0, pie_der0, 0.0)
    postura_izq, postura_der = np.array(postura_izq), np.array(postura_der)
    ultimo_izq = np.zeros(7)
    ultimo_der = np.zeros(7)
    max_paso_rad = VEL_MAX_SOFT / 60.0   # límite de cambio por ciclo de control

    tiempo_inicio = time.time()

    # Inicializar el agrupador de Sync Write (Dirección 116, longitud 4 bytes)
    groupSyncWrite = GroupSyncWrite(
        portHandler, packetHandler, ADDR_GOAL_POSITION, 4)

    try:
        contador = 0
        while True:
            if viewer is not None and not viewer.is_running():
                print("Ventana de MuJoCo cerrada.")
                break

            t_total = time.time() - tiempo_inicio

            if t_total < T_AGACHARSE:
                # --- A.0 Arranque: de pierna recta (offsets capturados) a la postura de pie ---
                s = t_total / T_AGACHARSE
                s = s * s * (3 - 2 * s)                       # suavizado
                solucion_izq = s * postura_izq
                solucion_der = s * postura_der
            else:
                # --- A. Generador de Trayectorias (pies planos, LIPM) ---
                t = t_total - T_AGACHARSE
                # Rampa de intensidad: primeros ciclos con pasos y balanceo crecientes
                intensidad = min(1.0, t / (CICLOS_RAMPA * params.periodo))
                target_izq, yaw_izq, target_der, yaw_der = generador.objetivos(t, intensidad)

                # --- B. Cinemática Inversa ---
                solucion_izq, solucion_der = resolver_ik(target_izq, yaw_izq, target_der, yaw_der)

            # Limitador de seguridad: ningún servo cambia más de max_paso_rad por ciclo
            solucion_izq = ultimo_izq + np.clip(np.array(solucion_izq) - ultimo_izq, -max_paso_rad, max_paso_rad)
            solucion_der = ultimo_der + np.clip(np.array(solucion_der) - ultimo_der, -max_paso_rad, max_paso_rad)
            ultimo_izq, ultimo_der = solucion_izq, solucion_der

            # --- C. Inyección a MuJoCo ---
            for i in range(6):
                data.qpos[indices_qpos_izq[i]] = solucion_izq[i + 1]
                data.qpos[indices_qpos_der[i]] = solucion_der[i + 1]

            mujoco.mj_kinematics(model, data)
            if viewer is not None:
                viewer.sync()

            # --- D. Inyección a Hardware Físico (SYNC WRITE) ---

            # Limpiar los parámetros del paquete anterior
            groupSyncWrite.clearParam()

            # 1. Empaquetar posiciones de la Pierna Izquierda
            for i, servo_id in enumerate(SERVO_IDS_IZQ):
                if servo_id in centros_fisicos_izq:
                    rads = solucion_izq[i + 1]
                    pasos = rad_a_dynamixel(
                        rads, centros_fisicos_izq[servo_id], config_giro_izq[servo_id])

                    # Convertir el entero a una matriz de 4 bytes (Little Endian)
                    pasos_formateados = pasos & 0xFFFFFFFF
                    param_goal_position = list(
                        pasos_formateados.to_bytes(4, byteorder='little'))

                    # Añadir al paquete maestro
                    groupSyncWrite.addParam(servo_id, param_goal_position)

            # 2. Empaquetar posiciones de la Pierna Derecha
            for i, servo_id in enumerate(SERVO_IDS_DER):
                if servo_id in centros_fisicos_der:
                    rads = solucion_der[i + 1]
                    pasos = rad_a_dynamixel(
                        rads, centros_fisicos_der[servo_id], config_giro_der[servo_id])

                    # Convertir el entero a una matriz de 4 bytes (Little Endian)
                    pasos_formateados = pasos & 0xFFFFFFFF
                    param_goal_position = list(
                        pasos_formateados.to_bytes(4, byteorder='little'))

                    # Añadir al paquete maestro
                    groupSyncWrite.addParam(servo_id, param_goal_position)

            # 3. Disparar el paquete sincronizado al bus (Un solo envío para todos los motores)
            groupSyncWrite.txPacket()

            # Imprime el timestamp y el ciclo para monitorizar los FPS
            contador += 1
            if contador % 60 == 0:
                print("%.3f : %d" % (time.time(), contador))
            # Mantener 60 Hz descontando lo que tardó la IK
            t_siguiente = tiempo_inicio + contador / 60.0
            espera = t_siguiente - time.time()
            if espera > 0:
                time.sleep(espera)

    except KeyboardInterrupt:
        print("\nEjecución detenida manualmente por el usuario.")
    finally:
        print("\nApagando torque de TODOS los servos instantáneamente (Broadcast)...")
        # El Broadcast ID 254 funciona perfectamente incluso si faltan motores en el bus
        packetHandler.write1ByteTxOnly(portHandler, 254, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.1)

        portHandler.closePort()

        if viewer is not None:
            viewer.close()
            print("Visualizador cerrado.")

        print("Fin del programa. Sistema liberado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Control de marcha del humanoide")
    parser.add_argument('--visualizar', action='store_true',
                        help="Abre la ventana 3D de MuJoCo")
    parser.add_argument('--periodo', type=float, default=1.6, help="s por ciclo (2 pasos)")
    parser.add_argument('--paso', type=float, default=2.0, help="cm de avance por paso")
    parser.add_argument('--altura', type=float, default=2.0, help="cm de altura del pie")
    parser.add_argument('--doble-apoyo', type=float, default=0.2)
    parser.add_argument('--zcom', type=float, default=20.0, help="cm, altura del CoM sobre el suelo")
    parser.add_argument('--xcom', type=float, default=0.0, help="cm, corrimiento X del CoM")
    parser.add_argument('--balanceo', type=float, default=1.0, help="ganancia del balanceo lateral")
    parser.add_argument('--giro', type=float, default=0.0, help="grados de giro por ciclo")
    args = parser.parse_args()

    iniciar_robot(visualizar=args.visualizar, params=ParamsMarcha(
        periodo=args.periodo, largo_paso=args.paso, altura_paso=args.altura,
        doble_apoyo=args.doble_apoyo, z_com=args.zcom, x_com=args.xcom,
        ganancia_balanceo=args.balanceo, giro_por_ciclo=args.giro))
