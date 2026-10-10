"""
Control manual de las piernas desde la terminal (funciona por SSH).

  Pierna IZQUIERDA          Pierna DERECHA
    W / S : X  +/-            I / K : X  +/-     (adelante / atrás)
    A / D : Y  +/-            J / L : Y  +/-     (izquierda / derecha)
    E / Q : Z  +/-            O / U : Z  +/-     (subir / bajar)

  +/- : paso más grande / más chico      ESPACIO : volver a la postura de pie
  P   : imprimir posiciones y ángulos    X o Ctrl+C : salir (apaga torque)

Las coordenadas son las del punto IK del pie en el frame base_link (cm):
+X adelante, +Y izquierda, +Z arriba. Los pies se mantienen planos.

Uso:
  python control_manual.py                  # con servos
  python control_manual.py --sin-servos --visualizar   # solo simulación MuJoCo
"""
import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

import test_servos_kin as robot
from marcha import ParamsMarcha, GeneradorMarcha

TECLAS = {
    # tecla: (pierna, eje, signo)   pierna 0 = izq, 1 = der; eje 0=X 1=Y 2=Z
    'w': (0, 0, +1), 's': (0, 0, -1), 'a': (0, 1, +1), 'd': (0, 1, -1), 'e': (0, 2, +1), 'q': (0, 2, -1),
    'i': (1, 0, +1), 'k': (1, 0, -1), 'j': (1, 1, +1), 'l': (1, 1, -1), 'o': (1, 2, +1), 'u': (1, 2, -1),
}
PASOS = [0.1, 0.25, 0.5, 1.0, 2.0]          # cm por pulsación
ERROR_MAX_IK = 0.05                          # cm: si la IK no llega, se rechaza el movimiento
LIMITE_RELATIVO = np.array([6.0, 6.0, 6.0])  # cm máximos de alejamiento respecto a la postura de pie
FPS = 60


class Teclado:
    """Lee teclas sin esperar Enter y sin bloquear (modo cbreak)."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.viejo = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.viejo)

    def leer(self):
        teclas = []
        while select.select([sys.stdin], [], [], 0)[0]:
            c = os.read(self.fd, 1).decode(errors="ignore")
            if c:
                teclas.append(c.lower())
        return teclas


def resolver(objetivos):
    """IK de ambas piernas. Devuelve (sol_izq, sol_der, ok)."""
    prev = (list(robot.angulos_previos_izq), list(robot.angulos_previos_der))
    sol_izq, sol_der = robot.resolver_ik(objetivos[0], 0.0, objetivos[1], 0.0)
    err = max(np.linalg.norm(robot.pierna_izquierda.forward_kinematics(sol_izq)[:3, 3] - objetivos[0]),
              np.linalg.norm(robot.pierna_derecha.forward_kinematics(sol_der)[:3, 3] - objetivos[1]))
    if err > ERROR_MAX_IK:
        robot.angulos_previos_izq, robot.angulos_previos_der = prev
        return None, None, False
    return sol_izq, sol_der, True


def estado(objetivos, base, paso, msg=""):
    rel = objetivos - base
    linea = (f"\rIZQ x{objetivos[0][0]:6.2f} y{objetivos[0][1]:6.2f} z{objetivos[0][2]:7.2f} | "
             f"DER x{objetivos[1][0]:6.2f} y{objetivos[1][1]:6.2f} z{objetivos[1][2]:7.2f} | "
             f"Δizq({rel[0][0]:+.1f},{rel[0][1]:+.1f},{rel[0][2]:+.1f}) "
             f"Δder({rel[1][0]:+.1f},{rel[1][1]:+.1f},{rel[1][2]:+.1f}) | paso {paso}cm {msg:<18}")
    sys.stdout.write(linea)
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Control manual XYZ de las piernas")
    ap.add_argument('--sin-servos', action='store_true', help="no abre el bus Dynamixel")
    ap.add_argument('--visualizar', action='store_true', help="muestra el modelo en MuJoCo")
    ap.add_argument('--z', type=float, default=-23.0, help="altura inicial del pie bajo la cadera (cm)")
    args = ap.parse_args()

    # Postura de pie = la misma que usa la marcha (intensidad 0)
    gen = GeneradorMarcha(ParamsMarcha(z_cadera=args.z), robot.neutro_izq, robot.neutro_der)
    pie_izq, _, pie_der, _ = gen.objetivos(0.0, intensidad=0.0)
    base = np.array([pie_izq, pie_der])
    objetivos = base.copy()
    postura_izq, postura_der, ok = resolver(objetivos)
    if not ok:
        print("La postura inicial no tiene solución IK. Revisa --z.")
        return

    conexion = None
    if not args.sin_servos:
        conexion = robot.conectar_servos()
        if conexion is None:
            return
        portHandler, packetHandler, centros_izq, centros_der = conexion
        groupSyncWrite = robot.GroupSyncWrite(portHandler, packetHandler, robot.ADDR_GOAL_POSITION, 4)

    viewer = None
    if args.visualizar:
        viewer = robot.mujoco.viewer.launch_passive(robot.model, robot.data)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 60.0, -20.0, 45.0
        viewer.cam.lookat[:] = [0.0, 0.0, -12.5]

    print(__doc__.split("Las coordenadas")[0])
    deseado_izq, deseado_der = postura_izq, postura_der
    ultimo_izq, ultimo_der = np.zeros(7), np.zeros(7)
    max_paso_rad = robot.VEL_MAX_SOFT / FPS * 0.5     # más lento que la marcha: es para probar
    i_paso = 1
    t0 = time.time()
    cuadro = 0
    msg = "agachándose..."

    try:
        with Teclado() as teclado:
            while True:
                if viewer is not None and not viewer.is_running():
                    break
                # Arranque suave: los primeros segundos solo se lleva el robot a la postura de pie
                listo = time.time() - t0 > robot.T_AGACHARSE
                if listo and msg == "agachándose...":
                    msg = "listo"

                for c in (teclado.leer() if listo else []):
                    if c == 'x':
                        raise KeyboardInterrupt
                    elif c in TECLAS:
                        pierna, eje, signo = TECLAS[c]
                        nuevo = objetivos.copy()
                        nuevo[pierna][eje] += signo * PASOS[i_paso]
                        if np.any(np.abs(nuevo - base) > LIMITE_RELATIVO):
                            msg = "límite de seguridad"
                            continue
                        si, sd, ok = resolver(nuevo)
                        if ok:
                            objetivos, deseado_izq, deseado_der, msg = nuevo, si, sd, "ok"
                        else:
                            msg = "fuera de alcance"
                    elif c in '+=':
                        i_paso = min(i_paso + 1, len(PASOS) - 1)
                    elif c in '-_':
                        i_paso = max(i_paso - 1, 0)
                    elif c == ' ':
                        objetivos = base.copy()
                        deseado_izq, deseado_der, _ = resolver(objetivos)
                        msg = "postura de pie"
                    elif c == 'p':
                        sys.stdout.write("\n")
                        for nom, sol in (("IZQ", deseado_izq), ("DER", deseado_der)):
                            print(nom, "ángulos (°):", np.round(np.degrees(np.array(sol)[1:]), 1),
                                  "[cad_roll, cad_pitch, rodilla, tob_pitch, tob_yaw, tob_roll]")

                # Seguir al objetivo con velocidad limitada (movimiento suave aunque el salto sea grande)
                ultimo_izq = ultimo_izq + np.clip(np.array(deseado_izq) - ultimo_izq, -max_paso_rad, max_paso_rad)
                ultimo_der = ultimo_der + np.clip(np.array(deseado_der) - ultimo_der, -max_paso_rad, max_paso_rad)

                for i in range(6):
                    robot.data.qpos[robot.indices_qpos_izq[i]] = ultimo_izq[i + 1]
                    robot.data.qpos[robot.indices_qpos_der[i]] = ultimo_der[i + 1]
                if viewer is not None:
                    robot.mujoco.mj_kinematics(robot.model, robot.data)
                    viewer.sync()
                if conexion is not None:
                    robot.enviar_angulos(groupSyncWrite, ultimo_izq, ultimo_der, centros_izq, centros_der)

                if cuadro % 6 == 0:
                    estado(objetivos, base, PASOS[i_paso], msg)
                cuadro += 1
                espera = t0 + cuadro / FPS - time.time()
                if espera > 0:
                    time.sleep(espera)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSaliendo...")
        if conexion is not None:
            packetHandler.write1ByteTxOnly(portHandler, 254, robot.ADDR_TORQUE_ENABLE, 0)
            time.sleep(0.1)
            portHandler.closePort()
            print("Torque apagado.")
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
