"""
Generador de marcha para UTABOT-Humanoid (reemplaza la sección "A. Generador de Trayectorias").

Idea general
------------
1. Se planifican los pies en el MUNDO (suelo): el pie de apoyo queda QUIETO en el suelo y
   el pie en vuelo avanza 2*L con un perfil de jerk mínimo (velocidad y aceleración 0 al
   despegar y al aterrizar -> no arrastra ni golpea).
2. Se define el ZMP de referencia: centro del pie de apoyo en apoyo simple, y transición
   lineal entre pies en doble apoyo.
3. El CoM se obtiene con el Péndulo Invertido Lineal (LIPM) resolviendo la solución
   PERIÓDICA exacta en frecuencia (FFT):  c'' = w0^2 (c - p)   ->   C_k = P_k * w0^2 / (w0^2 + (2*pi*k/T)^2)
   Así el balanceo lateral y el avance quedan sincronizados con los pasos y se ajustan
   solos a la velocidad (más rápido = menos balanceo, como en un humano).
4. Objetivo de cada pie en el frame de la cadera (base_link) = pie_mundo - cadera_mundo.

Convención (frame base_link del URDF, unidades en cm):
  +X = adelante (el pie se extiende 7 cm hacia +X y 4 cm hacia -X del tobillo)
  +Y = izquierda (cadera izquierda en y=+5.39)
  +Z = arriba
"""
from dataclasses import dataclass
import math
import numpy as np

G_CM = 981.0  # gravedad en cm/s^2


@dataclass
class ParamsMarcha:
    periodo: float = 1.2          # s, ciclo completo = 2 pasos (izq + der)
    largo_paso: float = 3.0       # cm, avance por paso (distancia entre pies al aterrizar)
    altura_paso: float = 2.0      # cm, altura máx. del pie en vuelo
    z_cadera: float = -23.0       # cm, altura del tobillo bajo la cadera (Z_REPOSO)
    doble_apoyo: float = 0.2      # fracción de cada medio ciclo con ambos pies en el suelo
    z_com: float = 20.0           # cm, altura estimada del CoM sobre el suelo (para LIPM)
    x_com: float = 0.0            # cm, posición X del CoM real respecto a base_link (calibrar)
    ganancia_balanceo: float = 1.0  # multiplica el balanceo lateral calculado (afinar 0.8-1.3)
    separacion_extra: float = 0.0   # cm, abre (+) o cierra (-) los pies lateralmente
    giro_por_ciclo: float = 0.0   # grados de yaw por ciclo (+ = gira a la izquierda)
    zmp_x_pie: float = -1.49      # cm, centro de la planta respecto al punto IK (del URDF)
    # --- Compensación de cedencia por gravedad (feedforward) ---
    comp_cadera_roll: float = 0.0  # grados extra de roll de cadera HACIA AFUERA en la pierna de apoyo
    comp_tobillo_roll: float = 0.0 # grados extra de roll de tobillo en la pierna de apoyo (mismo sentido)
    abrir_vuelo: float = 0.0      # cm que el pie en vuelo se abre hacia afuera a mitad del paso
    muestras: int = 512           # resolución de la tabla del ciclo


def _jerk_min(s):
    """Interpolación 0->1 con vel. y acel. nulas en los extremos."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _campana(s):
    """0 -> 1 -> 0 suave (vel. nula en extremos) para la altura del pie."""
    s = np.clip(s, 0.0, 1.0)
    return 0.5 * (1 - np.cos(2 * np.pi * s))


def _lipm_periodico(e, periodo, z_com):
    """Solución periódica de c'' = w0^2 (c - e) muestreada sobre un ciclo."""
    w0_2 = G_CM / z_com
    n = len(e)
    k = np.fft.rfftfreq(n, d=1.0 / n)          # armónicos 0..n/2
    wk = 2 * np.pi * k / periodo
    return np.fft.irfft(np.fft.rfft(e) * w0_2 / (w0_2 + wk**2), n)


class GeneradorMarcha:
    def __init__(self, p: ParamsMarcha, neutro_izq, neutro_der):
        """neutro_*: posición (x, y, z) del punto IK del pie con la pierna recta (FK con ángulos 0)."""
        self.p = p
        self.y_izq = neutro_izq[1] + p.separacion_extra / 2
        self.y_der = neutro_der[1] - p.separacion_extra / 2
        # postura de pie: centro de la planta bajo el CoM
        self.x0 = p.x_com - p.zmp_x_pie
        self._tabular()

    def _tabular(self):
        p = self.p
        n = p.muestras
        f = np.arange(n) / n
        h = 0.5
        ds = p.doble_apoyo * h
        L = p.largo_paso

        # Ventanas de vuelo: izquierda en el 1er medio ciclo, derecha en el 2do
        ini_i, fin_i = ds / 2, h - ds / 2
        ini_d, fin_d = h + ds / 2, 1 - ds / 2
        si = (f - ini_i) / (fin_i - ini_i)
        sd = (f - ini_d) / (fin_d - ini_d)
        vuela_i = (f >= ini_i) & (f <= fin_i)
        vuela_d = (f >= ini_d) & (f <= fin_d)

        # Pies en el MUNDO (X). En f=0: der adelante (+L/2), izq atrás (-L/2)
        xi = -L / 2 + 2 * L * _jerk_min(si)
        xd = +L / 2 + 2 * L * _jerk_min(sd)
        zi = np.where(vuela_i, p.altura_paso * _campana(si), 0.0)
        zd = np.where(vuela_d, p.altura_paso * _campana(sd), 0.0)

        # ZMP de referencia
        cx = p.zmp_x_pie
        px = np.empty(n)
        py = np.empty(n)
        for j, fj in enumerate(f):
            if fj < ini_i:                  # DS: der -> ... transición hacia el der (que soporta)
                a = (fj + ds / 2) / ds      # 0.5..1 dentro de la DS que cruza f=0
                px[j] = (1 - a) * (xi[j] + cx) + a * (xd[j] + cx)
                py[j] = (1 - a) * self.y_izq + a * self.y_der
            elif fj <= fin_i:               # SS sobre pie derecho
                px[j], py[j] = xd[j] + cx, self.y_der
            elif fj < ini_d:                # DS: der -> izq
                a = (fj - fin_i) / ds
                px[j] = (1 - a) * (xd[j] + cx) + a * (xi[j] + cx)
                py[j] = (1 - a) * self.y_der + a * self.y_izq
            elif fj <= fin_d:               # SS sobre pie izquierdo
                px[j], py[j] = xi[j] + cx, self.y_izq
            else:                           # DS: izq -> der (cierra el ciclo)
                a = (fj - fin_d) / ds       # 0..0.5
                px[j] = (1 - a) * (xi[j] + cx) + a * (xd[j] + cx)
                py[j] = (1 - a) * self.y_izq + a * self.y_der

        # CoM con LIPM periódico. En X separamos la rampa de avance 2*L*f
        avance = 2 * L * f
        cx_per = _lipm_periodico(px - avance, p.periodo, p.z_com)
        com_x = avance + cx_per
        com_y = _lipm_periodico(py, p.periodo, p.z_com) * p.ganancia_balanceo

        self.f = f
        self.tab = dict(xi=xi, xd=xd, zi=zi, zd=zd, com_x=com_x, com_y=com_y,
                        px=px, py=py, si=si, sd=sd, vuela_i=vuela_i, vuela_d=vuela_d)
        # objetivos relativos a la cadera (base_link)
        self.rel = dict(
            xi=xi - com_x + p.x_com, xd=xd - com_x + p.x_com,
            yi=self.y_izq - com_y + p.abrir_vuelo * np.where(vuela_i, np.sin(np.pi * np.clip(si, 0, 1))**2, 0.0),
            yd=self.y_der - com_y - p.abrir_vuelo * np.where(vuela_d, np.sin(np.pi * np.clip(sd, 0, 1))**2, 0.0),
            zi=p.z_cadera + zi, zd=p.z_cadera + zd,
        )

    def _interp(self, arr, fase):
        n = len(arr)
        u = (fase % 1.0) * n
        i = int(u) % n
        a = u - int(u)
        return (1 - a) * arr[i] + a * arr[(i + 1) % n]

    def objetivos(self, t, intensidad=1.0):
        """
        t: tiempo en s desde que empezó la marcha. intensidad: 0..1 (rampa de arranque/parada).
        Devuelve (pos_izq, yaw_izq, pos_der, yaw_der) en el frame base_link (cm, rad).
        """
        p = self.p
        fase = t / p.periodo
        r = self.rel
        k = intensidad
        xi = self.x0 + k * (self._interp(r["xi"], fase) - self.x0)
        xd = self.x0 + k * (self._interp(r["xd"], fase) - self.x0)
        yi = self.y_izq + k * (self._interp(r["yi"], fase) - self.y_izq)
        yd = self.y_der + k * (self._interp(r["yd"], fase) - self.y_der)
        zi = p.z_cadera + k * (self._interp(r["zi"], fase) - p.z_cadera)
        zd = p.z_cadera + k * (self._interp(r["zd"], fase) - p.z_cadera)

        # Giro: cada pie rota +-g/2 en vuelo y vuelve lineal en apoyo (aprox. para giros pequeños)
        yaw_i = yaw_d = 0.0
        if p.giro_por_ciclo:
            g = math.radians(p.giro_por_ciclo) * k
            fi = fase % 1.0
            yaw_i = self._yaw_pie(fi, g, 0.0)
            yaw_d = self._yaw_pie(fi, g, 0.5)
        return np.array([xi, yi, zi]), yaw_i, np.array([xd, yd, zd]), yaw_d

    def _yaw_pie(self, f, g, desfase):
        h = 0.5
        ds = self.p.doble_apoyo * h
        ini, fin = desfase + ds / 2, desfase + h - ds / 2
        f = f % 1.0
        if ini <= f <= fin:                               # vuelo: -g/2 -> +g/2
            return -g / 2 + g * float(_jerk_min((f - ini) / (fin - ini)))
        apoyo = 1 - (fin - ini)                           # apoyo: +g/2 -> -g/2 lineal
        s = ((f - fin) % 1.0) / apoyo
        return g / 2 - g * s

    def carga(self, t, intensidad=1.0):
        """
        Fracción del peso sobre cada pie (0..1), según dónde está el ZMP entre ambos pies.
        Sirve para escalar la compensación de gravedad: 1 en apoyo simple, transición en doble apoyo.
        Devuelve (carga_izq, carga_der).
        """
        py = self._interp(self.tab["py"], t / self.p.periodo)
        w_izq = float(np.clip((py - self.y_der) / (self.y_izq - self.y_der), 0.0, 1.0))
        # Con intensidad < 1 (rampa) se reparte hacia 50/50
        w_izq = 0.5 + intensidad * (w_izq - 0.5)
        return w_izq, 1.0 - w_izq

    def compensacion_gravedad(self, t, intensidad=1.0):
        """
        Offsets articulares (rad) a SUMAR después de la IK, por pierna, en orden de la cadena ikpy
        [base, cad_roll, cad_pitch, rodilla, tob_pitch, tob_yaw, tob_roll].

        Bajo carga, el roll de cadera de la pierna de apoyo cede y la pelvis cae hacia el lado del
        pie en vuelo; ese pie baja y se cruza hacia adentro. Se manda el roll "de más" hacia afuera
        en proporción a la carga para que, al ceder, quede donde debía.
        En el frame del URDF un roll + mueve el pie hacia -Y en ambas piernas:
          afuera izq (+Y) = roll negativo, afuera der (-Y) = roll positivo.
        Solo se cuenta la carga por ENCIMA del 50%: en doble apoyo parejo no se compensa nada.
        """
        w_izq, w_der = self.carga(t, intensidad)
        exceso_i = max(0.0, 2 * w_izq - 1)
        exceso_d = max(0.0, 2 * w_der - 1)
        cad = math.radians(self.p.comp_cadera_roll)
        tob = math.radians(self.p.comp_tobillo_roll)
        off_i = np.zeros(7)
        off_d = np.zeros(7)
        off_i[1] = -cad * exceso_i
        off_d[1] = +cad * exceso_d
        off_i[6] = -tob * exceso_i
        off_d[6] = +tob * exceso_d
        return off_i, off_d

    def resumen(self):
        p = self.p
        v = 2 * p.largo_paso / p.periodo
        return (f"T={p.periodo:.2f}s  paso={p.largo_paso:.1f}cm  vel={v:.1f}cm/s  "
                f"cadencia={120 / p.periodo:.0f} pasos/min  "
                f"balanceo CoM=±{np.max(np.abs(self.tab['com_y'])):.2f}cm")
