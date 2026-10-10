"""
Valida la marcha OFFLINE (sin servos): corre la IK de ikpy sobre un ciclo completo y reporta
error de IK, rango articular, velocidad y aceleración máximas por servo.

Uso:
  python validar_marcha.py                     # barrido de periodos/largos
  python validar_marcha.py --periodo 1.0 --paso 3 --graficar
"""
import argparse
import math
import warnings
import numpy as np
from ikpy.chain import Chain
from marcha import ParamsMarcha, GeneradorMarcha

warnings.filterwarnings("ignore")
URDF = "Humanoid/urdf/Humanoid.urdf"
NOMBRES_IZQ = ["Cuerpo1:12_link_joint", "Cuerpo1:8_link_joint", "Cuerpo1:6_link_joint",
               "Cuerpo1:4_link_joint", "Cuerpo1:3_link_joint", "Cuerpo1:1_link_joint"]
NOMBRES_DER = ["Cuerpo1:25_link_joint", "Cuerpo1:21_link_joint", "Cuerpo1:19_link_joint",
               "Cuerpo1:17_link_joint", "Cuerpo1:16_link_joint", "Cuerpo1:14_link_joint"]
ARTIC = ["cad_roll", "cad_pitch", "rodilla", "tob_pitch", "tob_yaw", "tob_roll"]

# XL430-W250 @ 11.1 V: ~57 rpm sin carga. Con carga real usar ~70% como margen
VEL_MAX = 57 * 2 * math.pi / 60 * 0.7       # rad/s
# Límites articulares de seguridad (rad): rodilla SIEMPRE doblada hacia adelante
LIMITES = [(-0.6, 0.6), (-1.6, 1.6), (-2.4, -0.05), (-1.6, 1.6), (-0.8, 0.8), (-0.6, 0.6)]


def cargar_pierna(nombres):
    c = Chain.from_urdf_file(
        URDF, base_elements=["base_link"] + [x for par in zip(
            nombres, [n.replace("_joint", "") for n in nombres]) for x in par],
        active_links_mask=[False] + [True] * 6)
    for i, lim in enumerate(LIMITES):
        c.links[i + 1].bounds = lim
    return c


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


SEMILLA = [0.0, 0.0, 0.4, -0.8, 0.4, 0.0, 0.0]   # rodilla adelante, igual que el branch actual


def simular(p: ParamsMarcha, fps=60, ciclos=2):
    izq, der = cargar_pierna(NOMBRES_IZQ), cargar_pierna(NOMBRES_DER)
    fk_i, fk_d = izq.forward_kinematics([0.0] * 7), der.forward_kinematics([0.0] * 7)
    gen = GeneradorMarcha(p, fk_i[:3, 3], fk_d[:3, 3])
    prev_i, prev_d = list(SEMILLA), list(SEMILLA)
    qs, errs = [], []
    n = int(ciclos * p.periodo * fps)
    for k in range(n):
        t = k / fps
        ti, yi, td, yd = gen.objetivos(t)
        Ri, Rd = rot_z(yi) @ fk_i[:3, :3], rot_z(yd) @ fk_d[:3, :3]
        si = izq.inverse_kinematics(target_position=ti, target_orientation=Ri,
                                    orientation_mode="all", initial_position=prev_i)
        sd = der.inverse_kinematics(target_position=td, target_orientation=Rd,
                                    orientation_mode="all", initial_position=prev_d)
        prev_i, prev_d = si, sd
        errs.append(max(np.linalg.norm(izq.forward_kinematics(si)[:3, 3] - ti),
                        np.linalg.norm(der.forward_kinematics(sd)[:3, 3] - td)))
        oi, od = gen.compensacion_gravedad(t)   # offsets intencionales: van después del chequeo de error
        qs.append(np.concatenate([(si + oi)[1:], (sd + od)[1:]]))
    q = np.array(qs)[int(p.periodo * fps):]       # descarta el 1er ciclo (arranque IK)
    dq = np.gradient(q, 1 / fps, axis=0)
    ddq = np.gradient(dq, 1 / fps, axis=0)
    return gen, q, dq, ddq, max(errs[int(p.periodo * fps):])


def reporte(p):
    gen, q, dq, ddq, err = simular(p)
    vmax = np.abs(dq).max(0)
    ok = err < 0.05 and vmax.max() < VEL_MAX
    print(f"{gen.resumen()} | err IK {err:.3f}cm | vel máx {vmax.max():.2f} rad/s "
          f"({ARTIC[vmax.argmax() % 6]}) | acel máx {np.abs(ddq).max():.0f} rad/s² | "
          f"{'OK' if ok else 'NO'}")
    return gen, q, dq, ok


def graficar(gen, q, fps=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tb, r, f = gen.tab, gen.rel, gen.f
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(f, r["xi"], label="pie izq X"); ax[0, 0].plot(f, r["xd"], label="pie der X")
    ax[0, 0].plot(f, r["zi"] - gen.p.z_cadera, "--", label="pie izq Z (altura)")
    ax[0, 0].plot(f, r["zd"] - gen.p.z_cadera, "--", label="pie der Z (altura)")
    ax[0, 0].set_title("Pies respecto a la cadera (cm)"); ax[0, 0].set_xlabel("fase del ciclo")
    ax[0, 1].plot(f, tb["py"], label="ZMP ref Y"); ax[0, 1].plot(f, tb["com_y"], label="CoM Y (LIPM)")
    ax[0, 1].axhspan(gen.y_izq - 3.95, gen.y_izq + 3.95, color="C0", alpha=.08)
    ax[0, 1].axhspan(gen.y_der - 3.95, gen.y_der + 3.95, color="C1", alpha=.08)
    ax[0, 1].set_title("Balanceo lateral (cm) — franjas = ancho de cada pie")
    ax[1, 0].plot(f, tb["px"] - 2 * gen.p.largo_paso * f, label="ZMP ref X (rel. avance)")
    ax[1, 0].plot(f, tb["com_x"] - 2 * gen.p.largo_paso * f, label="CoM X (rel. avance)")
    ax[1, 0].set_title("Avance (cm)"); ax[1, 0].set_xlabel("fase del ciclo")
    tt = np.arange(len(q)) / fps / gen.p.periodo
    for j in range(6):
        ax[1, 1].plot(tt, np.degrees(q[:, j]), label=f"izq {['cad_roll','cad_pitch','rodilla','tob_pitch','tob_yaw','tob_roll'][j]}")
    ax[1, 1].set_title("Ángulos pierna izquierda (°)"); ax[1, 1].set_xlabel("ciclos")
    for a in ax.flat:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(gen.resumen())
    fig.tight_layout(); fig.savefig("marcha.png", dpi=110)
    print("Gráfico guardado en marcha.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--periodo", type=float)
    ap.add_argument("--paso", type=float)
    ap.add_argument("--altura", type=float, default=2.0)
    ap.add_argument("--zcom", type=float, default=20.0)
    ap.add_argument("--graficar", action="store_true")
    a = ap.parse_args()
    if a.periodo:
        p = ParamsMarcha(periodo=a.periodo, largo_paso=a.paso or 3.0,
                         altura_paso=a.altura, z_com=a.zcom)
        gen, q, dq, ok = reporte(p)
        if a.graficar:
            graficar(gen, q)
    else:
        for T in [1.6, 1.2, 1.0, 0.8, 0.6]:
            for L in [2.0, 3.0, 4.0, 5.0]:
                reporte(ParamsMarcha(periodo=T, largo_paso=L, altura_paso=a.altura, z_com=a.zcom))
