#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banco del arg bogie_drop.  Sin ROS corriendo, sin Gazebo: solo xacro.

bogie_drop BAJA el eje de los pivotes del bogie (rev_3 / rev_4) para acortar
la palanca K = h_huella / r con la que el par de rueda enrolla el balancin.
Se hace en dos sitios que TIENEN que moverse juntos:

  * el origin de rev_3 / rev_4         baja  bogie_drop
  * los brazos del bogie (fusionados en Acople_Rodamiento) suben bogie_drop

Si alguien toca uno y no el otro, el modelo sigue generando y sigue
spawneando: lo que pasa es que las seis ruedas se hunden o se levantan
bogie_drop metros respecto al chasis, la altura de reposo cambia y con ella
el spawn_z.  Es el fallo silencioso que este banco caza.

Lo que comprueba, en orden de importancia:

1. LAS RUEDAS NO SE MUEVEN.  Con bogie_drop = 0.20 las seis ruedas, en el
   marco de base_link, tienen que quedar donde estaban al micrometro.  Lo
   mismo los brazos del balancin (-0.189 media / +0.237 trasera), que son lo
   que fija el momento de las cargas.
2. EL PIVOTE SI SE MUEVE, y exactamente lo pedido.
3. bogie_drop = 0.0 REPRODUCE EL MODELO DE REFERENCIA byte a byte, para que
   las corridas ya tomadas sigan siendo comparables.
4. EL IMU SIGUE AL CENTRO DE MASAS.  Bajar el pivote baja los dos soportes
   Acople_Rodamiento_*_1, y con ellos el CM del vehiculo: 2.94 mm por cada
   0.1 m de drop.  La tolerancia de brazo del IMU es 0.002 m
   (sim_config.yaml: imu_lever_arm_tolerance_m), asi que a partir de
   drop = 0.068 el IMU se sale del CM si no se le mueve con el.  Es lo mismo
   que comprueba check_sim_parity 13b, pero sin levantar Gazebo.
5. rocker_drop ACORTA LAS DOS RAMAS DEL ROCKER A LA VEZ.  Es el fallo
   simetrico: si una rama se acorta y la otra no, las seis ruedas dejan de
   ser coplanarias y el vehiculo se apoya en cuatro sin que nada avise.  Con
   el las ruedas SI se mueven -suben rocker_drop en el marco de base_link-,
   asi que lo que se comprueba es que suban las seis lo mismo, que nada se
   mueva en x ni en y, que los sensores conserven su altura sobre la huella
   y que spawn_z baje lo mismo.
6. LAS DOS RUTAS DE SPAWN PASAN LOS ARGS.  robot_description y
   swept_robot_description salen del mismo xacro con args distintos; si el
   arg falta en una, el modo swept spawnea otra suspension.  Mismo fallo que
   ya costo una campana con differential_linkage.

    python3 test_bogie_drop.py
"""
import io
import math
import os
import re
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
XACRO = os.path.join(RAIZ, 'src', 'rocker_bogie', 'urdf', 'ensamblajeurdf.xacro')
LAUNCH = os.path.join(RAIZ, 'src', 'rocker_bogie', 'launch',
                      'lcmine_rocker_bogie_world.launch')

RUEDAS = ['Rueda_1_1', 'Rueda_2_1', 'Rueda_3_1', 'Rueda_4_1', 'Rueda_5_1', 'Rueda_6_1']
PIVOTES = ['Acople_Rodamiento_1_1', 'Acople_Rodamiento_2_1']
TOL = 1e-9

fallos = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FALLA ') + msg)
    if not cond:
        fallos.append(msg)


def genera(drop, rocker=None):
    """URDF plano con bogie_drop=drop, con el resto de args como el launch.
    rocker=None deja rocker_drop en su default, que es lo que se spawnea."""
    cmd = ['xacro', '--inorder', XACRO,
           'differential:=false', 'differential_linkage:=true',
           'bogie_upper:=0.6', 'bogie_drop:=%s' % drop]
    if rocker is not None:
        cmd.append('rocker_drop:=%s' % rocker)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.exit('xacro fallo con bogie_drop=%s rocker_drop=%s:\n%s'
                 % (drop, rocker, p.stderr.decode()))
    return p.stdout.decode()


def arbol(urdf):
    """{link_hijo: (link_padre, (x, y, z))} de todos los joints."""
    t = {}
    for m in re.finditer(r'<joint name="([^"]+)"[^>]*>(.*?)</joint>', urdf, re.S):
        cuerpo = m.group(2)
        o = re.search(r'<origin[^>]*xyz="([^"]*)"', cuerpo)
        p = re.search(r'<parent link="([^"]+)"', cuerpo)
        h = re.search(r'<child link="([^"]+)"', cuerpo)
        if not (o and p and h):
            continue
        if h.group(1) == 'chassis_link':
            continue   # el banco mide en el marco del chasis; base_link es el marco centrado
        t[h.group(1)] = (p.group(1), tuple(float(v) for v in o.group(1).split()))
    return t


def pos(t, link, extra=None):
    """Posicion del link en base_link, a deflexion cero.  Todos los rpy de
    esta cadena son cero, asi que basta con sumar.  extra es un offset dentro
    del propio link (el origen inercial, para el CM)."""
    xyz = [0.0, 0.0, 0.0] if extra is None else list(extra)
    visto = set()
    while link in t and link not in visto:
        visto.add(link)
        padre, o = t[link]
        xyz = [a + b for a, b in zip(xyz, o)]
        link = padre
    return tuple(xyz)


TOL_IMU = 0.002   # sim_config.yaml: sensor_mounting.imu_lever_arm_tolerance_m


def masas(urdf):
    """{link: (masa, origen inercial)} de los links que pesan."""
    L = {}
    for m in re.finditer(r'<link name="([^"]+)">(.*?)</link>', urdf, re.S):
        n, b = m.group(1), m.group(2)
        ms = re.search(r'<mass value="([^"]*)"', b)
        # (?:<!--.*?-->|\s)* : este xacro documenta dentro de <inertial>,
        # y exigir que <origin> fuese lo primero dejaba el origen inercial en
        # (0,0,0) sin avisar -> el CM salia 20 cm bajo y el banco culpaba al IMU
        o = re.search(r'<inertial>(?:<!--.*?-->|\s)*<origin[^>]*xyz="([^"]*)"',
                      b, re.S)
        if ms and float(ms.group(1)) > 0:
            L[n] = (float(ms.group(1)),
                    tuple(float(v) for v in o.group(1).split()) if o else (0.0, 0.0, 0.0))
    return L


def centro_de_masas(t, L, sin='imu_link'):
    """CM del vehiculo SIN un link.  El IMU se monta en el CM del resto, que es
    la solucion cerrada de p = c(p)."""
    M = 0.0
    S = [0.0, 0.0, 0.0]
    for n, (m, c) in L.items():
        if n == sin:
            continue
        q = pos(t, n, c)
        M += m
        S = [a + m * b for a, b in zip(S, q)]
    return tuple(v / M for v in S)


print('1. las ruedas y los brazos NO se mueven al bajar el pivote')
# 0.10 y no 0.20: con rocker_drop en su defecto el pivote del bogie sale de
# 0.178 m sobre el eje, asi que 0.20 lo meteria por debajo del plano de las
# ruedas.  El tope vive en la nota de bogie_drop del xacro y lo comprueba la
# seccion 2.
DROP = 0.10
t0, t1 = arbol(genera(0.0)), arbol(genera(DROP))
for w in RUEDAS:
    a, b = pos(t0, w), pos(t1, w)
    d = max(abs(x - y) for x, y in zip(a, b))
    check(d < TOL, '%-10s quieta (desvio %.2e m)' % (w, d))

for piv, med, tra in [('Acople_Rodamiento_2_1', 'Rueda_3_1', 'Rueda_5_1'),
                      ('Acople_Rodamiento_1_1', 'Rueda_4_1', 'Rueda_2_1')]:
    for w in (med, tra):
        a = pos(t0, w)[0] - pos(t0, piv)[0]
        b = pos(t1, w)[0] - pos(t1, piv)[0]
        check(abs(a - b) < TOL,
              'brazo %s->%-10s sigue en %+.4f m' % (piv, w, b))

print('\n2. el pivote SI baja, y lo justo')
for piv in PIVOTES:
    dz = pos(t0, piv)[2] - pos(t1, piv)[2]
    check(abs(dz - DROP) < TOL, '%s baja %.4f m (pedido %.4f)' % (piv, dz, DROP))
    dxy = max(abs(pos(t0, piv)[i] - pos(t1, piv)[i]) for i in (0, 1))
    check(dxy < TOL, '%s no se mueve en x ni en y' % piv)

# la palanca, que es para lo que existe el arg
R = 0.178
h0 = pos(t0, 'Acople_Rodamiento_2_1')[2] - pos(t0, 'Rueda_3_1')[2] + R
h1 = pos(t1, 'Acople_Rodamiento_2_1')[2] - pos(t1, 'Rueda_3_1')[2] + R
print('     K = h_huella/r : %.2f -> %.2f  (%.0f %% menos)'
      % (h0 / R, h1 / R, 100 * (1 - h1 / h0)))
check(h1 < h0, 'la palanca se acorta')
check(h1 > R - TOL, 'la palanca no baja de r = %.3f m (la huella siempre esta '
                    'un radio por debajo del eje)' % R)

# EL TOPE DE bogie_drop LO FIJA rocker_drop.  Es la interaccion entre los dos
# args, y es silenciosa: pasarse no da error, solo mete el eje del bogie por
# debajo del plano de las ruedas y K vuelve a subir por el otro lado.
_xac = io.open(XACRO, newline='').read()
_rd = float(re.search(r'<xacro:arg name="rocker_drop" default="([^"]*)"',
                      _xac).group(1))
TOPE_BOGIE = 0.351444 - max(0.0, _rd - 0.170395)
print('     tope de bogie_drop con rocker_drop %.6f: %.6f m' % (_rd, TOPE_BOGIE))
check(DROP <= TOPE_BOGIE,
      'el drop que prueba este banco (%.3f) cabe en el tope' % DROP)
_alto = pos(arbol(genera(TOPE_BOGIE)), 'Acople_Rodamiento_2_1')[2] \
    - pos(arbol(genera(TOPE_BOGIE)), 'Rueda_3_1')[2]
check(abs(_alto) < 1e-5,   # 1 um del redondeo a 6 cifras del reescalado
      'en el tope el pivote del bogie queda en el plano de los ejes '
      '(%.2e m), o sea K = 1.00' % _alto)

print('\n3. el default del arg es lo que se spawnea, y xacro y launch coinciden')
xac = io.open(XACRO, newline='').read()
def_xacro = re.search(r'<xacro:arg name="bogie_drop" default="([^"]*)"', xac).group(1)
def_launch = re.search(r'<arg name="bogie_drop" default="([^"]*)"',
                       io.open(LAUNCH, newline='').read()).group(1)
check(float(def_xacro) == float(def_launch),
      'default del xacro %s == default del launch %s' % (def_xacro, def_launch))
sin_arg = subprocess.run(['xacro', '--inorder', XACRO,
                          'differential:=false', 'differential_linkage:=true',
                          'bogie_upper:=0.6'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
check(sin_arg == genera(def_xacro),
      'bogie_drop:=%s reproduce el modelo por defecto byte a byte' % def_xacro)
check(genera(def_xacro) != genera(DROP), 'y otro valor da otro modelo (el arg llega)')

print('\n4. el IMU sigue al centro de masas')
for d in (0.0, 0.060, 0.120, 0.178):
    u = genera(d)
    tt, LL = arbol(u), masas(u)
    imu = pos(tt, 'imu_link')
    cm = centro_de_masas(tt, LL)
    brazo = sum((a - b) ** 2 for a, b in zip(imu, cm)) ** 0.5
    check(brazo < TOL_IMU,
          'drop %.3f -> IMU z %+.6f, CM z %+.6f, brazo %.2f mm (tope %.0f)'
          % (d, imu[2], cm[2], 1000 * brazo, 1000 * TOL_IMU))

# y que de verdad baja lo que dice la cuenta cerrada: masa de los dos
# soportes por el drop, entre la masa del vehiculo sin el IMU.  Si alguien
# deja el z clavado, o mete otro coeficiente, esto lo caza.
MASS_SCALE = 0.474039
COEF = 2 * 1.39525 * MASS_SCALE / (45.0 - 0.0149159)
z0 = pos(arbol(genera(0.0)), 'imu_link')[2]
z1 = pos(arbol(genera(0.178)), 'imu_link')[2]
check(abs((z0 - z1) - COEF * 0.178) < 1e-9,
      'el IMU baja %.3f mm con drop 0.178 (cuenta cerrada %.3f)'
      % (1000 * (z0 - z1), 1000 * COEF * 0.178))

print('\n5. las dos rutas de spawn pasan el arg')
lan = io.open(LAUNCH, newline='').read()
check(re.search(r'<arg name="bogie_drop"\s+default="[0-9.]+"\s*/>', lan) is not None,
      'el launch declara bogie_drop')
cmds = re.findall(r'command="[^"]*ensamblajeurdf\.xacro[^"]*"', lan, re.S)
check(len(cmds) == 2, 'hay dos rutas de spawn (encontradas %d)' % len(cmds))
check(any('retopic_lidar.py' in c for c in cmds),
      'una de ellas es la del LiDAR barrido')
for i, c in enumerate(cmds):
    check('bogie_drop:=$(arg bogie_drop)' in c, 'ruta %d pasa bogie_drop' % (i + 1))

print('\n6. la huella reescalada')
AREA_TRACKED = 0.3996      # batalla x via del tracked, table_robot_specs
R_RUEDA = 0.148
tf = arbol(genera(def_xacro))
Q = {w: pos(tf, w) for w in RUEDAS}
# centro REAL de la rueda: la colision va 0.052315 m hacia fuera del origen
# del link.  Medir desde el origen dejo la huella un 19 % grande (2026-09-13).
_cy = sum(q[1] for q in Q.values()) / 6.0
Q = {w: (q[0], q[1] + math.copysign(0.052315, q[1] - _cy), q[2]) for w, q in Q.items()}
xy = [(q[0], q[1]) for q in Q.values()]
cx = sum(a for a, _ in xy) / 6.0
cy = sum(b for _, b in xy) / 6.0
orden = sorted(xy, key=lambda z: math.atan2(z[1] - cy, z[0] - cx))
area = 0.5 * abs(sum(orden[i][0] * orden[(i + 1) % 6][1]
                     - orden[(i + 1) % 6][0] * orden[i][1] for i in range(6)))
check(abs(area - AREA_TRACKED) / AREA_TRACKED < 0.005,
      'area de apoyo %.4f m2, a %.1f %% de la del tracked (%.4f)'
      % (area, 100 * abs(area - AREA_TRACKED) / AREA_TRACKED, AREA_TRACKED))
sep = min(abs(Q['Rueda_3_1'][0] - Q['Rueda_1_1'][0]),
          abs(Q['Rueda_5_1'][0] - Q['Rueda_3_1'][0]))
check(sep > 2 * R_RUEDA,
      'las ruedas no se tocan: %.4f m entre centros, %+.4f m de holgura'
      % (sep, sep - 2 * R_RUEDA))
zs = [q[2] for q in Q.values()]
# 1e-5 y no 0: el reescalado escribio los origins con 6 cifras significativas,
# asi que quedan ~1 um.  Despreciable contra los 90 um que se hunde el contacto.
check(max(zs) - min(zs) < 1e-5,
      'las seis ruedas siguen coplanarias en z %+.4f (dispersion %.1e m, '
      'redondeo del reescalado)' % (zs[0], max(zs) - min(zs)))

print('\n7. el bogie es una pieza de tubos rectos, solo dibujo')
# 2026-09-13: las piezas fijas del bogie se fusionaron en Acople_Rodamiento.
# Se vigila que siga SIN colision con cualquier drop y que su brazo recto al
# pivote de direccion trasero mida lo que dicen las juntas.
for d in (0.0, 0.089, 0.178):
    u = genera(d)
    tt = arbol(u)
    for pieza, dire in (('Acople_Rodamiento_1_1', 'BarraRuedas_3_1'),
                        ('Acople_Rodamiento_2_1', 'BarraRuedas_6_1')):
        trozo = re.search(r'<link name="%s">(.*?)</link>' % pieza, u, re.S).group(1)
        tubos = [float(v) for v in re.findall(r'<cylinder[^>]*length="([0-9.eE+-]+)"', trozo)]
        check(tt[dire][0] == pieza, 'drop %.3f %s cuelga de %s' % (d, dire, tt[dire][0]))
        brazo = math.sqrt(sum(v ** 2 for v in tt[dire][1]))
        check(any(abs(v - brazo) < 1e-6 for v in tubos),
              'drop %.3f %s: brazo recto de %.4f m al pivote de direccion' % (d, pieza, brazo))
        check('<collision' not in trozo,
              'drop %.3f %s sigue SIN colision (es solo dibujo)' % (d, pieza))

print('\n8. rocker_drop: acortar el brazo del rocker')
# rocker_drop acorta las DOS ramas del rocker en la misma cantidad.  El fallo
# silencioso de aqui es el simetrico del de bogie_drop: si una rama se acorta
# y la otra no, las seis ruedas dejan de ser coplanarias y el vehiculo se
# apoya en cuatro sin que nada avise.
RD_TECHO = 0.170395 + 0.211702      # rama trasera + montante comun
r_xacro = float(re.search(r'<xacro:arg name="rocker_drop" default="([^"]*)"',
                          io.open(XACRO, newline='').read()).group(1))
lan8 = io.open(LAUNCH, newline='').read()
r_launch = float(re.search(r'<arg name="rocker_drop" default="([^"]*)"',
                           lan8).group(1))
check(r_xacro == r_launch,
      'default del xacro %.6f == default del launch %.6f' % (r_xacro, r_launch))
check(0.0 <= r_xacro <= RD_TECHO + 1e-9,
      'el default %.6f cae dentro del techo %.6f (rama trasera 0.170395 mas '
      'montante 0.211702)' % (r_xacro, RD_TECHO))
# En el techo el montante es cero y link_r_27 se queda sin longitud, asi que
# su tubo se degenera.  El default tiene que dejar material.
_post = 1.0 - max(0.0, r_xacro - 0.170395) / 0.211702
check(_post > 0.05,
      'el montante conserva el %.1f %% de su altura (link_r_27 en %.1f mm)'
      % (100 * _post, 1000 * 0.07905 * _post))

for rd in (0.0, 0.08, 0.170395, r_xacro):
    ta, tb = arbol(genera(0.0, 0.0)), arbol(genera(0.0, rd))
    subidas = [pos(tb, w)[2] - pos(ta, w)[2] for w in RUEDAS]
    lat = max(abs(pos(tb, w)[i] - pos(ta, w)[i]) for w in RUEDAS for i in (0, 1))
    check(max(abs(v - rd) for v in subidas) < 1e-6,
          'rocker_drop %.6f: las seis ruedas suben %.6f m' % (rd, subidas[0]))
    check(max(subidas) - min(subidas) < 1e-6,
          'rocker_drop %.6f: y suben LO MISMO (dispersion %.1e m, siguen '
          'coplanarias)' % (rd, max(subidas) - min(subidas)))
    check(lat < 1e-9,
          'rocker_drop %.6f: ninguna rueda se mueve en x ni en y (%.1e m)'
          % (rd, lat))

# los dos pivotes a la misma altura: es lo que define el techo
tt = arbol(genera(0.0))
z_rock = pos(tt, 'AcopleTuboRB_1_1')[2]
z_bog = pos(tt, 'Acople_Rodamiento_1_1')[2]
z_eje = pos(tt, 'Rueda_4_1')[2]
print('     pivotes sobre el eje: rocker %+.6f  bogie %+.6f  (K %.2f y %.2f)'
      % (z_rock - z_eje, z_bog - z_eje,
         (z_rock - z_eje + 0.178) / 0.178, (z_bog - z_eje + 0.178) / 0.178))
if r_xacro >= 0.170395 - 1e-9:
    check(abs(z_rock - z_bog) < 1e-6,
          'pasado el primer tramo los dos pivotes van nivelados (%.1e m)'
          % abs(z_rock - z_bog))

# los sensores conservan su altura SOBRE LA HUELLA, que es lo que iguala las
# tres plataformas; el montaje sube con el chasis y por eso lleva mastil
for sensor in ('velodyne_base_link', 'camera_link'):
    hs = []
    for rd in (0.0, 0.08, 0.170395, r_xacro):
        tc = arbol(genera(0.0, rd))
        hs.append(pos(tc, sensor)[2] - (pos(tc, 'Rueda_4_1')[2] - 0.178))
    check(max(hs) - min(hs) < 1e-6,
          '%s se queda a %.4f m de la huella con cualquier rocker_drop '
          '(dispersion %.1e)' % (sensor, hs[0], max(hs) - min(hs)))

# el IMU sigue al CM tambien aqui.  Acortar el brazo sube mas de media masa
# del vehiculo, muchisimo mas que los 2.94 mm/0.1 m de bogie_drop.
for rd in (0.0, 0.08, 0.170395, r_xacro):
    u = genera(0.0, rd)
    tc, LL = arbol(u), masas(u)
    brazo = sum((a - b) ** 2 for a, b in
                zip(pos(tc, 'imu_link'), centro_de_masas(tc, LL))) ** 0.5
    check(brazo < TOL_IMU,
          'rocker_drop %.6f -> brazo del IMU %.2f mm (tope %.0f)'
          % (rd, 1000 * brazo, 1000 * TOL_IMU))

# spawn_z va emparejado con el RADIO, no con rocker_drop: base_link va a la
# altura de los ejes, que es un radio sobre la huella, asi que bajar la rueda
# baja el spawn lo mismo.  Esperado = 1.2148 - (0.178 - r), con el 1.2148 que
# valia cuando la rueda medida 0.178.  Escrito derivado y no a mano, que es
# lo que hizo fallar este check al pasar la rueda a 0.148.
Z_SPAWN_R178 = 1.2148
z_esperado = Z_SPAWN_R178 - (0.178 - R_RUEDA)
z_spawn = float(re.search(r'<arg name="spawn_z" default="([^"]*)"', lan8).group(1))
check(abs(z_spawn - z_esperado) < 0.011,
      'spawn_z %.4f == %.4f (= 1.2148 - (0.178 - r), r = %.3f): base_link va '
      'a la altura de los ejes, asi que sigue al radio y no a rocker_drop'
      % (z_spawn, z_esperado, R_RUEDA))
for i, c in enumerate(re.findall(r'command="[^"]*ensamblajeurdf\.xacro[^"]*"',
                                 lan8, re.S)):
    check('rocker_drop:=$(arg rocker_drop)' in c,
          'ruta %d pasa rocker_drop' % (i + 1))

print('\n9. formas simples: ninguna malla salvo las seis ruedas')
# Los eslabones estructurales se dibujan con tubos y cajas.  Una malla nueva
# aqui es un eslabon que ha vuelto al CAD, y el CAD no se reescalo con la
# huella: se solaparia en pantalla y dejaria de seguir a rocker_drop.
u = genera(0.0)
mallas = re.findall(r'<mesh filename="([^"]*)"', u)
check(len(mallas) == 6, 'quedan %d mallas (deberian ser 6)' % len(mallas))
check(all('/Rueda_' in m for m in mallas),
      'y las seis son de rueda: %s' % sorted(set(os.path.basename(m) for m in mallas)))

# Y que cada tubo LLEGUE de verdad a la junta que dice.  Un tubo se dibuja
# entre dos puntos escritos a mano: si uno se teclea mal, el eslabon sigue
# generando y sigue pesando lo mismo, solo que en pantalla queda un brazo
# apuntando al aire y una junta flotando.  Lo que se exige es que el origen
# de CADA joint hijo caiga en la punta de algun cilindro del padre.
EXENTOS = {
    # leafs sin hijos: no hay junta a la que llegar
    'AcopleRB_1_1', 'AcopleRB_2_1',
    # el motor se dibuja como su propia carcasa, no como un brazo a la rueda
    'Motor_1_1', 'Motor_2_1', 'Motor_3_1', 'Motor_4_1', 'Motor_5_1', 'Motor_6_1',
    # las ruedas conservan su malla
    'Rueda_1_1', 'Rueda_2_1', 'Rueda_3_1', 'Rueda_4_1', 'Rueda_5_1', 'Rueda_6_1',
    # el chasis es una caja mas dos mastiles de sensor, no un esqueleto
    'base_link', 'chassis_link',
    # sensores y varillaje del diferencial: forma propia
    'velodyne_base_link', 'imu_link', 'camera_link', 'camera_optical_frame',
    'differential_beam', 'diff_rod_left', 'diff_rod_right',
}


def puntas(trozo):
    """Los dos extremos de cada cilindro del link, en el marco del link."""
    out = []
    for m in re.finditer(r'<visual>(.*?)</visual>', trozo, re.S):
        v = m.group(1)
        # xacro escribe los atributos en orden alfabetico: rpy antes que xyz
        oxyz = re.search(r'<origin[^>]*\bxyz="([^"]*)"', v)
        orpy = re.search(r'<origin[^>]*\brpy="([^"]*)"', v)
        cil = re.search(r'<cylinder[^>]*length="([0-9.eE+-]+)"', v)
        if not (oxyz and orpy and cil):
            continue
        c = [float(t) for t in oxyz.group(1).split()]
        _, pitch, yaw = [float(t) for t in orpy.group(1).split()]
        L = float(cil.group(1))
        d = [math.sin(pitch) * math.cos(yaw) * L / 2.0,
             math.sin(pitch) * math.sin(yaw) * L / 2.0,
             math.cos(pitch) * L / 2.0]
        out.append([c[i] + d[i] for i in range(3)])
        out.append([c[i] - d[i] for i in range(3)])
    return out


u2 = genera(0.227)
hijos = {}
for m in re.finditer(r'<joint name="[^"]+"[^>]*>(.*?)</joint>', u2, re.S):
    cu = m.group(1)
    p = re.search(r'<parent link="([^"]+)"', cu)
    o = re.search(r'<origin[^>]*xyz="([^"]*)"', cu)
    if p and o:
        hijos.setdefault(p.group(1), []).append(
            [float(v) for v in o.group(1).split()])

sueltos = []
for m in re.finditer(r'<link name="([^"]+)">(.*?)</link>', u2, re.S):
    n, cuerpo = m.group(1), m.group(2)
    if n in EXENTOS or n not in hijos:
        continue
    pts = puntas(cuerpo)
    for h in hijos[n]:
        if not any(max(abs(a - b) for a, b in zip(h, q)) < 1e-6 for q in pts):
            sueltos.append('%s -> (%.4f %.4f %.4f)' % (n, h[0], h[1], h[2]))
check(not sueltos,
      'los %d eslabones con esqueleto llegan a todas sus juntas'
      % (len(hijos) - len(EXENTOS & set(hijos)))
      if not sueltos else 'juntas que no tocan ningun tubo: %s' % sueltos)

print('\n10. ninguna rueda pasa por debajo del chasis')
# Bajar el chasis solo es gratis si no le acerca ninguna rueda.  No se la
# acerca: la caja va de y -0.009 a 0.241 y las ruedas apoyan 0.23 m por fuera
# a cada lado.  Se comprueba articulando los cuatro pivotes pasivos a tope.
CAJA_X, CAJA_Y = (-0.055791, 0.544209), (0.0040, 0.1967)
SUSP = ['rocker_pivot_left', 'rocker_pivot_right', 'rev_3', 'rev_4']
R_LLANTA, SEMI = 0.178, 0.0572


def arbol_ejes(urdf):
    """Como arbol(), mas el eje de cada joint que tiene uno."""
    t = {}
    for m in re.finditer(r'<joint name="([^"]+)"[^>]*>(.*?)</joint>', urdf, re.S):
        nom, cuerpo = m.group(1), m.group(2)
        o = re.search(r'<origin[^>]*xyz="([^"]*)"', cuerpo)
        p = re.search(r'<parent link="([^"]+)"', cuerpo)
        h = re.search(r'<child link="([^"]+)"', cuerpo)
        a = re.search(r'<axis[^>]*xyz="([^"]*)"', cuerpo)
        if not (o and p and h):
            continue
        if h.group(1) == 'chassis_link':
            continue   # el banco mide en el marco del chasis; base_link es el marco centrado
        t[h.group(1)] = (nom, p.group(1),
                         tuple(float(v) for v in o.group(1).split()),
                         tuple(float(v) for v in a.group(1).split()) if a else None)
    return t


def pos_q(t, link, q):
    """Posicion del link con los joints de q girados.  Todos los pivotes que
    se barren aqui giran sobre +-Y, asi que basta con el seno y el coseno."""
    x, z = 0.0, 0.0
    y = 0.0
    visto = set()
    while link in t and link not in visto:
        visto.add(link)
        nom, padre, d, eje = t[link]
        if eje and nom in q and abs(abs(eje[1]) - 1.0) < 1e-6:
            a = q[nom] * (1.0 if eje[1] > 0 else -1.0)
            c, sn = math.cos(a), math.sin(a)
            x, z = c * x + sn * z, -sn * x + c * z
        x += d[0]
        y += d[1]
        z += d[2]
        link = padre
    return x, y, z


te = arbol_ejes(genera(0.0))
peor = None
rejilla = [-0.6 + 1.2 * i / 6.0 for i in range(7)]
# los rockers llegan a +-1.0 rad desde 2026-09-13; los bogies siguen en +-0.6
rejilla_rocker = [-1.0 + 2.0 * i / 6.0 for i in range(7)]
for a in rejilla_rocker:
    for b in rejilla_rocker:
        for c in rejilla:
            for e in rejilla:
                q = dict(zip(SUSP, (a, b, c, e)))
                for w in RUEDAS:
                    px, py, pz = pos_q(te, w, q)
                    if px + R_LLANTA < CAJA_X[0] or px - R_LLANTA > CAJA_X[1]:
                        continue
                    if py + SEMI < CAJA_Y[0] or py - SEMI > CAJA_Y[1]:
                        continue
                    hueco = 0.515 - (pz + R_LLANTA)
                    if peor is None or hueco < peor[0]:
                        peor = (hueco, w, q)
check(peor is None,
      'ninguna rueda entra bajo la caja del chasis en 2401 poses de la '
      'suspension' if peor is None else
      'la rueda %s se mete bajo el chasis: hueco %+.4f m' % (peor[1], peor[0]))

print('\n%s  (%d fallos)' % ('TODO BIEN' if not fallos else 'HAY FALLOS', len(fallos)))
sys.exit(1 if fallos else 0)
