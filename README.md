# UTABOT-Humanoid
A Humanoid Robot for HRR (Human Robot Racing) made by the UTABOT Team.
## Instalación
```bash
# 1. Clonar el repositorio y acceder al directorio
git clone https://github.com/MauriceTZ/UTABOT-Humanoid.git
cd UTABOT-Humanoid/

# 2. Crear el entorno virtual (heredando paquetes del sistema)
python -m venv .venv --system-site-packages

# 3. Activar el entorno virtual
source .venv/bin/activate

# 4. Instalar las dependencias
pip install -r requirements.txt

```

## Ejecución

Asegúrate de tener el entorno virtual activado (`source .venv/bin/activate`) cada vez que abras una nueva terminal.

Para iniciar el programa principal, ejecuta:

```bash
python test_servos_kin.py

```

## Control manual de las piernas

Mueve el pie de cada pierna en XYZ desde la terminal (funciona por SSH):

```bash
python control_manual.py                          # con servos
python control_manual.py --sin-servos --visualizar  # solo simulación
```

| Pierna izquierda | Pierna derecha | Movimiento |
|---|---|---|
| W / S | I / K | X adelante / atrás |
| A / D | J / L | Y izquierda / derecha |
| E / Q | O / U | Z subir / bajar |

`+`/`-` cambian el tamaño del paso, `ESPACIO` vuelve a la postura de pie, `P` imprime los ángulos y `X` sale (apaga el torque).
