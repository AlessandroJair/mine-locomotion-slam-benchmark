#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dos figuras 2-D del rocker-bogie.

    python3 rocker_2d.py --out ~/figs

  1. rocker_2d_dimensiones.eps  - alzado lateral acotado (plano x-z).
  2. rocker_2d_escalon.eps      - subida cuasi-estatica de un escalon de 0.2 m.

FORMATO DE PAPER.  El texto de las figuras va en INGLES y ninguna lleva
titulo: en un paper el titulo es el pie, asi que los pies se imprimen en la
consola al generar, y ahi van tambien los escalares y las medidas
transversales, que en un alzado lateral no se pueden acotar.  Los comentarios
del codigo se quedan en castellano, como el resto del paquete.

EL ALZADO ACOTA LA COLISION, que es la que simula y la que el estudio iguala
a mano en las tres plataformas.  Hasta el 2026-09-16 eso NO era lo mismo que
la malla: el visual de la rueda iba a la escala del export de CAD
(0.002588 0.002126 0.002329), que sobre un STL de 86 x 50 x 86 unidades
dejaba un ovalo de 111.3 x 100.1 mm de semieje contra un cilindro de 178 de
radio, o sea la rueda un 44 % pequeña -por eso en Gazebo no llegaba al chasis
y el vehiculo parecia flotar- y ovalada.  Corregido en el xacro a
0.00413953 0.002286 0.00413953, recalculando los seis <origin> del visual,
que es el fallo silencioso: en URDF el vertice cae en origin + escala * v,
asi que cambiar la escala DESPLAZA la malla.  El script sigue comprobandolo
en cada pasada y avisa si vuelve a descuadrar.

LA VISTA FRONTAL SE QUITO a peticion.  Lo que solo se veia alli -las dos
vias, los anchos y el varillaje del diferencial- se imprime en el pie.

LA GEOMETRIA NO SE ESCRIBE AQUI.  Sale del xacro, pasando por
suspension_sim.genera_urdf, y se transforma al marco de base_link con el
propio joint base_to_chassis.  Asi que cambiar rocker_drop o bogie_drop mueve
las dos figuras solo; ninguna cota esta copiada a mano.

EL MECANISMO PLANO.  Todas las juntas pasivas giran sobre +-Y, asi que un
lado es un mecanismo plano: pivote del rocker en el chasis, un brazo a la
rueda delantera y otro al pivote del bogie, y del bogie dos brazos a la media
y a la trasera.  Es la misma reduccion que hace suspension_sim.

POR QUE EL ROCKER NO ARTICULA EN LA FIGURA DEL ESCALON.  El diferencial impone
phi_izq + phi_der = 0 (es la restriccion que resuelve suspension_sim), y un
escalon que cruza todo el ancho mueve los dos rockers lo MISMO, asi que
phi_izq = phi_der y las dos cosas juntas dan phi = 0: el rocker queda rigido
con el chasis y es el CHASIS el que cabecea.  Lo que traga el desnivel es el
balancin mas el cabeceo, que es el comportamiento de libro.  Con el escalon
bajo una sola oruga habria que resolver los dos lados a la vez, y eso es
suspension_sim, no esto.

CUASI-ESTATICO Y DETERMINADO.  Con las tres ruedas tocando, el sistema tiene
tres incognitas (zc, cabeceo, balancin) contra tres huelgos nulos: se
RESUELVE, no se optimiza.  La misma nota de suspension_sim explica por que
minimizar energia potencial con desigualdades da ramas imposibles.  El huelgo
es la distancia del centro de la rueda a la FRONTERA del terreno menos el
radio, asi que una rueda apoyada en la arista del escalon sale gratis.

LAS LLANTAS SE VEN ENTRE SI.  Cada rueda apoya en el terreno O en otra
llanta, la que encuentre primero.  Hizo falta porque con las ruedas de 0.178
el modelo devolvia poses con las llantas metidas una en otra hasta 46.6 mm:
en reposo solo quedaban 11.7 mm entre llantas del mismo lado, y el brazo del
balancin al eje medio esta inclinado 53.7 grados -el pivote va 222 mm por
encima de los ejes- asi que girarlo para SUBIR la rueda media la echa hacia
DELANTE y se los comia.  Con 0.178 el vehiculo se ATASCABA: subia 29 mm de
escalon y nada mas.

CON LA RUEDA A 0.148 (2026-09-16, bajada en las tres plataformas justo por
esto) ya no se tocan en ningun punto del recorrido:

    bogie  -34.4 deg -> holgura front-mid  +228.6 mm
    bogie    0.0     ->                     +71.7
    bogie  +20.0     ->                     +13.4
    bogie  +34.4     ->                      +4.7

lo mas justo de todo el barrido son 3.8 mm, y el resultado se da la vuelta:

    el escalon de 200 mm SE SUBE ENTERO, con 13.4 mm de holgura minima entre
    la delantera y la media.  El maximo que sube es 296 mm, o sea DOS VECES
    el radio de rueda, que es la cifra que se le supone a un rocker-bogie.

Lo que sigue en pie de aquello: en el modelo self_collide va en false, asi
que Gazebo NUNCA calcula contacto rueda-rueda y no habria protestado por la
interpenetracion.  Lo dijo la nota de ensamblajeurdf.xacro.damp_orig
-"STILL TO CHECK: whether the CAD linkage actually clears itself"- y con
0.178 la respuesta era no.  Con 0.148 es si, pero sigue sin haber red: si
alguien vuelve a subir el radio o a ampliar el tope del bogie, nada en la
simulacion avisara.  Este script si: imprime la holgura peor en cada pasada.

LO QUE NO ES.  No hay fuerzas: ni par de rueda, ni la palanca K que segun la
nota de bogie_drop es lo que de verdad levanta las ruedas.  Y el rocker va
BLOQUEADO por el diferencial, que es lo que toca con un escalon que cruza
todo el ancho -el peor caso- pero no con uno bajo una sola oruga: ahi los dos
rockers giran y hay que resolver los dos lados a la vez, y eso es
suspension_sim, no esto.
"""
from __future__ import print_function

import argparse
import math
import os
import struct
import sys
import tempfile

import numpy as np
from scipy.optimize import root

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_parser as mp                                # noqa: E402
import suspension_sim as ss                              # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Circle, Rectangle, Polygon  # noqa: E402

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'lines.linewidth': 1.0, 'axes.grid': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

COLUMN_WIDTH_IN = 6.5
COTA = 'k'
MM = True          # un plano se acota en milimetros; --metros lo cambia
R = ss.R_RUEDA
ESCALON = 0.20
# la rueda izquierda de cada eje, que es la que se ve en la vista lateral
LADO = {'front': 'Rueda_6_1', 'mid': 'Rueda_4_1', 'rear': 'Rueda_2_1'}


# --------------------------------------------------------------- geometria
def semiejes_malla(geo):
    """Semiejes de la caja envolvente de una malla, ya escalada.

    Hace falta para poder AVISAR: lo que Gazebo pinta es el visual, y si no
    coincide con la colision el vehiculo se ve de un tamaño y se simula de
    otro.  STL binario: cuatro bytes con el numero de triangulos y 50 por
    triangulo, de los que los 36 utiles son los tres vertices."""
    if geo is None or geo.get('type') != 'mesh':
        return None
    ruta = os.path.join(ss.RAIZ, 'src', 'rocker_bogie', 'meshes',
                        os.path.basename(geo['uri']))
    if not os.path.exists(ruta):
        return None
    with open(ruta, 'rb') as f:
        d = f.read()
    n = struct.unpack('<I', d[80:84])[0]
    v = np.frombuffer(d[84:84 + n * 50], dtype=np.uint8)
    v = v.reshape(n, 50)[:, 12:48].copy().view('<f4').reshape(-1, 3)
    return 0.5 * (v.max(0) - v.min(0)) * np.asarray(geo['scale'])


def modelo(**args):
    """Puntos del mecanismo plano, en el marco de base_link (avance = +x,
    origen en el centro de la huella y a la altura de los ejes).

    La cinematica directa la hace model_parser, que compone las matrices
    completas.  Sumar los <origin> a mano NO vale: base_to_chassis lleva
    yaw = pi y las manguetas llevan rpy propio, asi que una suma de xyz
    saca las vias mal."""
    urdf = ss.genera_urdf(**args)
    fd, ruta = tempfile.mkstemp(suffix='.urdf')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(urdf)
        mod = mp.load_urdf(ruta)
    finally:
        os.unlink(ruta)
    L = mod.links

    def xz(nombre):
        p = L[nombre].pose[:3, 3]
        return np.array([p[0], p[2]])

    def y(nombre):
        return L[nombre].pose[1, 3]

    pares = {}
    for pref in ('AcopleTuboRB', 'Acople_Rodamiento'):
        pares[pref] = sorted([n for n in L if n.startswith(pref)], key=y)

    m = {'rocker': xz(pares['AcopleTuboRB'][0]),
         'bogie': xz(pares['Acople_Rodamiento'][0])}
    for cual, w in LADO.items():
        m[cual] = xz(w)
    for s in ('velodyne_base_link', 'camera_link', 'imu_link'):
        m[s] = xz(s)
    T, geo = L['chassis_link'].visuals[0]
    c = (L['chassis_link'].pose @ T)[:3, 3]
    m['chasis_c'] = np.array([c[0], c[2]])
    m['chasis_size'] = np.array([geo['size'][0], geo['size'][2]])
    m['chasis_ancho'] = geo['size'][1]
    m['masa'] = mod.link_mass()
    for j in mod.joints.values():
        if j.name == 'rev_3':
            m['lim_bogie'] = (j.limit['lower'], j.limit['upper'])
        elif j.name == 'rocker_pivot_left':
            m['lim_rocker'] = (j.limit['lower'], j.limit['upper'])
    # RADIO Y CENTRO REAL DE LA RUEDA.  El cilindro de colision va 0.052315 m
    # hacia FUERA del origen del link -solo en y-, asi que x y z valen tal
    # cual pero la via y la huella medidas desde el origen salen un 19 %
    # cortas.  Es el mismo ajuste que hace test_bogie_drop.py (seccion 6).
    Tw, gw = L['Rueda_4_1'].collisions[0]
    m['r'] = gw['radius']
    m['ancho_llanta'] = gw['length']
    cen = {}
    for w in ('Rueda_1_1', 'Rueda_2_1', 'Rueda_3_1',
              'Rueda_4_1', 'Rueda_5_1', 'Rueda_6_1'):
        p = (L[w].pose @ L[w].collisions[0][0])[:3, 3]
        cen[w] = np.array([p[0], p[1], p[2]])
    m['via_del'] = abs(cen['Rueda_1_1'][1] - cen['Rueda_6_1'][1])
    m['via_tras'] = abs(cen['Rueda_5_1'][1] - cen['Rueda_2_1'][1])
    # LA MALLA QUE GAZEBO DIBUJA NO ES LA RUEDA QUE SIMULA.  El visual es un
    # STL con escala propia y ANISOTROPA (0.002588 0.002126 0.002329), que
    # sobre una caja de 86 x 50 x 86 unidades deja un ovalo de 111.3 x 100.1
    # mm de semieje contra el cilindro de colision de 178.  O sea que en
    # pantalla la rueda sale un 44 % pequeña y no llega al chasis, pero la
    # fisica la usa de 178 y por eso el vehiculo parece flotar.  Este plano
    # acota la COLISION, que es la que simula y la que el estudio iguala a
    # mano en las tres plataformas; se guarda tambien la de la malla para
    # poder avisar de la diferencia.
    gm = L['Rueda_4_1'].visuals[0][1]
    m['malla_rueda'] = semiejes_malla(gm)
    m['malla_escala'] = np.asarray(gm.get('scale', (1.0, 1.0, 1.0)))
    xy = np.array([[c[0], c[1]] for c in cen.values()])
    o = xy[np.argsort(np.arctan2(xy[:, 1] - xy[:, 1].mean(),
                                 xy[:, 0] - xy[:, 0].mean()))]
    m['huella'] = 0.5 * abs(np.sum(o[:, 0] * np.roll(o[:, 1], -1)
                                   - np.roll(o[:, 0], -1) * o[:, 1]))
    # el rayo del Velodyne sale 0.0641 m por encima de su base (ver el
    # <sensor><pose> de ensamblajeurdf.gazebo)
    m['lidar_rayo'] = m['velodyne_base_link'][1] + 0.0641
    # LA VIGA DEL DIFERENCIAL Y SU VARILLA.  En el alzado lateral la viga se
    # ve DE CANTO -es un cilindro a lo ancho, sobre el eje de balanceo- asi
    # que es un circulo de su radio; la varilla si se ve entera, porque baja
    # inclinada 55 grados en este mismo plano, de la viga al soporte de
    # BarraBR.  Los dos lados coinciden en esta proyeccion.
    if 'differential_beam' in L:
        m['viga'] = xz('differential_beam')
        m['viga_r'] = L['differential_beam'].visuals[0][1]['radius']
        Tv, gv2 = L['diff_rod_left'].visuals[0]
        largo = gv2['length']
        p = (L['diff_rod_left'].pose
             @ np.array([0.0, 0.0, -largo, 1.0]))[:3]
        m['varilla_fin'] = np.array([p[0], p[2]])
        m['varilla_r'] = gv2['radius']
    return m


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s], [-s, c]])


def ruedas(m, xc, zc, theta, beta):
    """Centro de las tres ruedas del lado, en el mundo."""
    C, Rt = np.array([xc, zc]), rot(theta)
    piv = C + Rt.dot(m['rocker'])
    bp = C + Rt.dot(m['bogie'])
    Rb = Rt.dot(rot(beta))
    return {'front': C + Rt.dot(m['front']),
            'mid': bp + Rb.dot(m['mid'] - m['bogie']),
            'rear': bp + Rb.dot(m['rear'] - m['bogie'])}, piv, bp


# ----------------------------------------------------------------- terreno
def terreno(h=ESCALON, x_s=0.0, largo=3.0):
    """Frontera del solido como polilinea: llano, cara vertical, meseta."""
    return np.array([[-largo, 0.0], [x_s, 0.0], [x_s, h], [largo, h]])


def contacto(p, poli):
    """(distancia, punto de contacto) del punto a la polilinea.  La rueda
    nunca esta dentro del solido, asi que esa distancia ES la distancia al
    terreno, y apoyar en la ARISTA del escalon sale solo: el minimo lo da el
    extremo de un segmento."""
    a, b = poli[:-1], poli[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
    q = a + t[:, None] * ab
    d = np.hypot(*(p - q).T)
    k = int(np.argmin(d))
    return d[k], q[k]


def dist_frontera(p, poli):
    return contacto(p, poli)[0]


# ----------------------------------------------------- el arco de cada rueda
# QUE SE MANDA, Y POR QUE NO ES EL AVANCE DEL CHASIS.  Con un escalon mas
# alto que el radio (0.20 contra 0.178) ninguna rueda puede rodar sobre la
# arista: se queda APOYADA EN LA CARA VERTICAL y sube por ella empujada por
# las otras dos.  Mientras eso pasa su x esta clavada en -r, y como el pivote
# del rocker cae justo encima de la rueda media, el CHASIS NO AVANZA: mandar
# xc pide soluciones que no existen y el solver se va a una rama imposible
# (medido: cabeceo -59 deg, bogie +135, la media por debajo del suelo).
# Mandar el arco de la delantera falla por lo mismo en cuanto la media llega
# a la cara.
#
# LO QUE SI ES MONOTONO es la SUMA de los tres arcos: en todo instante hay al
# menos una rueda rodando hacia delante, asi que la suma crece siempre.  El
# arco de cada rueda se mide sobre el TERRENO DESPLAZADO un radio hacia
# fuera, que es el camino que recorre su centro mientras toca.


def camino(h=ESCALON, r=R, largo=2.0):
    """Terreno desplazado un radio: llano, cara, arista y meseta.

    Devuelve (f(s) -> centro de rueda, s(centro), longitud)."""
    if h >= r:                      # toca la CARA antes que la arista
        x_a, a0 = -r, math.pi
    else:                           # llega a la arista rodando por el llano
        a0 = math.pi - math.asin((r - h) / r)
        x_a = r * math.cos(a0)
    l1 = largo + x_a                # llano    z = r
    l2 = max(0.0, h - r)            # cara     x = x_a
    l3 = r * (a0 - math.pi / 2.0)   # arista   radio r sobre (0, h)
    c = np.cumsum([l1, l2, l3, largo])

    def f(s):
        if s <= c[0]:
            return np.array([-largo + s, r])
        if s <= c[1]:
            return np.array([x_a, r + (s - c[0])])
        if s <= c[2]:
            a = a0 - (s - c[1]) / r
            return np.array([r * math.cos(a), h + r * math.sin(a)])
        return np.array([s - c[2], h + r])

    def s_de(p, poli):
        """Arco de la rueda cuyo centro esta en p.

        El tramo NO se decide mirando p -el ruido de 1 um de la coplanaridad
        confundia el llano con la cara- sino el PUNTO DE CONTACTO, que es la
        misma cuenta que da el huelgo.  Los cuatro tramos empalman sin salto:
        en el paso llano->cara los dos dan c[0], y en cara->arista, c[1]."""
        q = contacto(p, poli)[1]
        tol = 1e-9
        if abs(q[0]) < tol and abs(q[1] - h) < tol:      # arista
            return c[1] + r * (a0 - math.atan2(p[1] - h, p[0]))
        if abs(q[1]) < tol:                              # llano
            return q[0] + largo
        if abs(q[0]) < tol:                              # cara
            return c[0] + q[1] - r
        return c[2] + q[0]                               # meseta

    return f, s_de, c[-1]


# LA MEDIA TIENE DOS CONTACTOS POSIBLES, Y HAY QUE ELEGIR UNO.  Con las tres
# ruedas mirando solo al terreno el modelo devolvia poses con las llantas
# metidas una en otra hasta 46.6 mm: en reposo solo hay 11.7 mm de holgura
# entre llantas de un mismo lado -es el minimo que fijo el reescalado x0.862
# de la huella- y el giro del balancin se los come.
#
# QUE PASA DE VERDAD: cuando la media llega a tocar la delantera, deja de
# apoyar en el suelo y sigue subiendo LLEVADA POR LA LLANTA DE LA DELANTERA.
# Sigue habiendo tres contactos, solo que uno es rueda-rueda, asi que el
# sistema sigue determinado y no hay que tocar nada mas.
#
# POR QUE NO CON min() DE LOS DOS HUELGOS, que es lo primero que se prueba:
# el residuo deja de ser derivable justo en el cambio, el jacobiano numerico
# salta y el solver se va (medido: residuo 1.2e+01 y cabeceos de 1e19 grados).
# Cada fase tiene su juego de ecuaciones FIJO y liso, y entre fases se
# conmuta mirando el signo del huelgo que sobra.
SUELO, RUEDA = 'suelo', 'rueda'


def residuos(m, poli, s_de, S, v, modo=SUELO):
    """Tres huelgos nulos mas el arco total mandado."""
    W, _, _ = ruedas(m, v[0], v[1], v[2], v[3])
    k = ('front', 'mid', 'rear')
    h = [dist_frontera(W[a], poli) - R for a in k]
    if modo == RUEDA:
        h[1] = math.hypot(*(W['mid'] - W['front'])) - 2 * R
    return h + [sum(s_de(W[a], poli) for a in k) - S]


def _sobra(m, poli, v, modo):
    """El huelgo que la fase NO impone.  Si se hace negativo, toca conmutar."""
    W, _, _ = ruedas(m, v[0], v[1], v[2], v[3])
    if modo == SUELO:
        return math.hypot(*(W['mid'] - W['front'])) - 2 * R
    return dist_frontera(W['mid'], poli) - R


def barrido(m, poli, s_de, S0, S1, paso=0.006, tol=1e-8):
    """Resuelve la pose en cada estacion.  Predictor lineal: extrapolar la
    solucion anterior es lo que mantiene el solver en su rama.

    EN CADA ESTACION SE PRUEBAN LAS DOS FASES, la de siempre primero para no
    romper la continuidad.  Una fase vale si converge Y deja el huelgo que
    NO impone en positivo.  No mirar la convergencia no basta: pasado el
    contacto entre llantas, la fase de suelo no se queda con el huelgo
    rueda-rueda negativo -se queda clavada en cero- y lo que se viola es su
    propia ecuacion, con la media metida hasta 51 mm en la cara del escalon.
    Si no vale ninguna, es un ATASCO de verdad y el barrido se corta ahi."""
    arcos = np.arange(S0, S1 + 0.5 * paso, paso)
    v = np.array([-1.5, R, 0.0, 0.0])
    ant, modo, out = None, SUELO, []
    for S in arcos:
        g = v if ant is None else 2.0 * v - ant
        elegida = None
        for cual in (modo, RUEDA if modo == SUELO else SUELO):
            u = root(lambda q: residuos(m, poli, s_de, S, q, cual), g,
                     method='hybr', options={'xtol': 1e-12}).x
            res = max(abs(np.array(residuos(m, poli, s_de, S, u, cual))))
            if res < tol and _sobra(m, poli, u, cual) >= -1e-9:
                elegida = (u, res, cual)
                break
        if elegida is None:
            return out, S
        ant, (v, res, modo) = v.copy(), elegida
        out.append((S, v.copy(), res, modo))
    return out, None


# ------------------------------------------------------------------ dibujo
def dibuja_robot(ax, m, xc, zc, theta, beta, relleno=True, sensores=True):
    """Alzado del mecanismo, a linea: visible continuo y grueso, ejes finos.

    MONOCROMO a proposito: un plano no lleva color."""
    W, piv, bp = ruedas(m, xc, zc, theta, beta)
    C, Rt = np.array([xc, zc]), rot(theta)
    # chasis
    hs = m['chasis_size'] / 2.0
    esq = [C + Rt.dot(m['chasis_c'] + np.array([sx * hs[0], sz * hs[1]]))
           for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(Polygon(esq, closed=True, lw=1.3, ec='k', zorder=2,
                         fc='0.94' if relleno else 'none'))
    # mastil y sensores
    if sensores:
        lid = C + Rt.dot(m['velodyne_base_link'])
        techo = C + Rt.dot(m['chasis_c'] + np.array([0.0, hs[1]]))
        ax.plot([lid[0], techo[0]], [lid[1], techo[1]], '-', color='k',
                lw=1.1, zorder=2)
        ax.add_patch(Circle(lid, 0.0387, fc='0.88', ec='k', lw=1.1, zorder=5))
        cam = C + Rt.dot(m['camera_link'])
        ax.add_patch(Rectangle(cam - 0.019, 0.038, 0.038, fc='0.88', ec='k',
                               lw=1.1, zorder=5))
    # brazos, por encima del chasis para que se lea el mecanismo
    for a, b in ((piv, W['front']), (piv, bp),
                 (bp, W['mid']), (bp, W['rear'])):
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='k', lw=2.2,
                solid_capstyle='round', zorder=6)
    # varillaje del diferencial: la viga de canto y su varilla.  Cuelga del
    # chasis, asi que gira con el
    if 'viga' in m:
        vg = C + Rt.dot(m['viga'])
        vf = C + Rt.dot(m['varilla_fin'])
        # EL MUÑON DEL ROCKER, que faltaba.  La varilla no cierra en el
        # pivote del rocker sino en el soporte de BarraBR (link_r_5/6), 50.9
        # mm delante y 53.2 debajo, y ese brazo ES la palanca del
        # diferencial: llevarlo al pivote la pone a cero y el diferencial
        # deja de acoplar los dos lados.  Sin dibujar el muñon, la varilla
        # acababa en el aire y el tramo final se leia como un sobrante.
        ax.plot([piv[0], vf[0]], [piv[1], vf[1]], '-', color='k', lw=2.2,
                solid_capstyle='round', zorder=6)
        # LINEA MAS FINA que los brazos, y es lo que le toca: la varilla es
        # de 8 mm de radio contra tubos de 20, y con el mismo grosor se leia
        # como una prolongacion del brazo del rocker, que lo cruza
        ax.plot([vg[0], vf[0]], [vg[1], vf[1]], '-', color='k', lw=1.1,
                solid_capstyle='round', zorder=6)
        ax.add_patch(Circle(vg, m['viga_r'], fc='w', ec='k', lw=1.1,
                            zorder=7))
        ax.plot([vg[0]], [vg[1]], '+', color='k', ms=4, zorder=8)
        # la union es una junta de BOLA, no un pivote: circulo RELLENO, que
        # se distingue del blanco-con-punto de las juntas de revolucion
        ax.add_patch(Circle(vf, 0.013, fc='k', ec='k', lw=1.0, zorder=10))
    # ruedas SIN relleno: si no, tapan los brazos del balancin
    for c in ('front', 'mid', 'rear'):
        ax.add_patch(Circle(W[c], R, fc='none', ec='k', lw=1.3, zorder=7))
        eje(ax, W[c])
    for p in (piv, bp):
        ax.add_patch(Circle(p, 0.017, fc='w', ec='k', lw=1.1, zorder=9))
        ax.plot([p[0]], [p[1]], '.', color='k', ms=2.5, zorder=10)
    return W, piv, bp


def eje(ax, p, l=0.045):
    """Marca de centro: raya y punto, como en un plano."""
    ax.plot([p[0] - l, p[0] + l], [p[1], p[1]], '-', color='k', lw=0.5,
            dashes=(7, 2, 1, 2), zorder=8)
    ax.plot([p[0], p[0]], [p[1] - l, p[1] + l], '-', color='k', lw=0.5,
            dashes=(7, 2, 1, 2), zorder=8)



def cot(v):
    """Una medida, en las unidades del plano.  Un plano se acota en mm."""
    return ('%.1f' % (1000.0 * v)) if MM else ('%.4f' % v)


# Lineas de referencia: hueco en la pieza y rebase por encima de la cota, que
# es como se dibuja una cota de verdad.  Sin el hueco la linea nace dentro del
# material y no se distingue de una arista.
HUECO, REBASE = 0.013, 0.016


def _aux(ax, p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', color=COTA, lw=0.4, zorder=1)


def ref_v(ax, x, z_pieza, z_cota):
    """Linea de referencia vertical, de la PIEZA a su cota.

    Hace falta una por extremo y no una por cota, porque los extremos de una
    cadena no estan a la misma altura: los brazos del balancin se miden entre
    dos centros de rueda (z = r) y el pivote del bogie, que esta 222 mm mas
    arriba.  Con un solo `ref` por cota, o la linea nacia donde no habia
    nada o no se dibujaba."""
    d = math.copysign(1.0, z_cota - z_pieza)
    _aux(ax, (x, z_pieza + d * HUECO), (x, z_cota + d * REBASE))


def cota_h(ax, x1, x2, z, txt, ref=None, dz=0.010, arriba=True):
    """Cota horizontal, con lineas de referencia desde ref."""
    ax.annotate('', (x1, z), (x2, z),
                arrowprops=dict(arrowstyle='<->', lw=0.6, color=COTA,
                                shrinkA=0, shrinkB=0))
    if ref is not None:
        d = math.copysign(1.0, z - ref)
        for x in (x1, x2):
            _aux(ax, (x, ref + d * HUECO), (x, z + d * REBASE))
    if txt:
        ax.text((x1 + x2) / 2.0, z + (dz if arriba else -dz), txt,
                ha='center', va='bottom' if arriba else 'top', color=COTA,
                fontsize=7.5)


def cota_v(ax, x, z1, z2, txt, ref=None, ha='left'):
    """Cota vertical.  El texto gira con la cota, que es lo normal en un
    plano (acotacion alineada, se lee de abajo arriba)."""
    ax.annotate('', (x, z1), (x, z2),
                arrowprops=dict(arrowstyle='<->', lw=0.6, color=COTA,
                                shrinkA=0, shrinkB=0))
    if ref is not None:
        d = math.copysign(1.0, x - ref)
        for z in (z1, z2):
            _aux(ax, (ref + d * HUECO, z), (x + d * REBASE, z))
    if txt:
        ax.text(x + (0.012 if ha == 'left' else -0.012), (z1 + z2) / 2.0, txt,
                ha=ha, va='center', color=COTA, fontsize=7.5, rotation=90)


def suelo(ax, x0, x1, z=0.0):
    """Linea de terreno y rayado, fino: en un plano es una seccion."""
    ax.plot([x0, x1], [z, z], '-', color='k', lw=1.0, zorder=1)
    ax.fill_between([x0, x1], z, z - 0.035, color='none', hatch='/////',
                    ec='0.55', lw=0.0, zorder=0)


def marco(ax, x0, x1, z0, z1, nota):
    """Cierra el alzado: sin ejes ni rejilla, y con la nota de unidades, que
    es lo unico de texto que lleva un plano."""
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.text(x1, z0, nota, fontsize=7, ha='right', va='bottom', color='0.3')


def fig_dimensiones(m):
    """Alzado lateral acotado, SOLO el mecanismo de suspension.

    FORMATO DE PLANO: sin ejes, sin leyenda, sin titulo y sin color; nada de
    texto salvo las medidas y la nota de unidades.  Lo que era leyenda (que
    pivote es cada uno, sus topes) se ha ido al pie de figura: un plano dice
    cuanto mide, no como se llama."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.5))
    W, piv, bp = dibuja_robot(ax, m, 0.0, R, 0.0, 0.0, relleno=False,
                              sensores=False)
    x0, x1, z0, z1 = -1.02, 1.02, -0.32, 0.66
    suelo(ax, x0, x1)

    # lo que sale de m esta en el marco de base_link, con el origen a la
    # altura de los EJES; el plano mide sobre la HUELLA, o sea un radio mas
    def hw(z):
        return z + R

    cz = hw(m['chasis_c'][1])
    largo_ch, alto_ch = m['chasis_size']
    panza, techo = cz - alto_ch / 2.0, cz + alto_ch / 2.0

    # --- cotas horizontales, en cadena debajo del suelo.  Las lineas de
    # referencia las pone ref_v, cada una desde SU pieza: los ejes desde el
    # centro de rueda y el pivote del bogie desde el pivote
    z1a, z2a, z3a = -0.075, -0.165, -0.255
    for a, b in ((W['rear'][0], W['mid'][0]), (W['mid'][0], W['front'][0])):
        cota_h(ax, a, b, z1a, cot(b - a), arriba=False)
    cota_h(ax, W['rear'][0], W['front'][0], z2a,
           cot(W['front'][0] - W['rear'][0]), arriba=False)
    # brazos del balancin: los dos tramos comparten linea
    cota_h(ax, W['rear'][0], bp[0], z3a, cot(bp[0] - W['rear'][0]),
           arriba=False)
    cota_h(ax, bp[0], W['mid'][0], z3a, cot(W['mid'][0] - bp[0]),
           arriba=False)
    for c in ('rear', 'mid', 'front'):
        ref_v(ax, W[c][0], W[c][1], z1a)
    for x, zp, zc in ((W['rear'][0], W['rear'][1], z3a),
                      (W['front'][0], W['front'][1], z2a),
                      (W['mid'][0], W['mid'][1], z3a),
                      (bp[0], bp[1], z3a)):
        ref_v(ax, x, zp, zc)
    # --- chasis, encima
    # +0.10 y no +0.055: la viga del diferencial asoma 38 mm por encima del
    # techo del chasis y la cota le pasaba por encima
    cota_h(ax, -largo_ch / 2.0, largo_ch / 2.0, techo + 0.10, cot(largo_ch),
           ref=techo)

    # --- cotas verticales, fuera de la silueta
    cota_v(ax, -0.60, 0.0, hw(piv[1] - R), cot(hw(piv[1] - R)), ref=bp[0])
    cota_v(ax, -0.76, 0.0, panza, cot(panza), ref=-largo_ch / 2.0)
    cota_v(ax, 0.60, panza, techo, cot(alto_ch), ref=largo_ch / 2.0,
           ha='right')
    # radio de la rueda, acotado desde el centro como manda el plano
    ax.annotate('', W['front'], W['front'] + np.array([R, 0.0]),
                arrowprops=dict(arrowstyle='-|>', lw=0.6, color=COTA,
                                shrinkA=0, shrinkB=0))
    ax.text(W['front'][0] + R / 2.0, W['front'][1] + 0.016, 'R' + cot(R),
            ha='center', color=COTA, fontsize=7.5,
            bbox=dict(fc='w', ec='none', pad=0.6))

    marco(ax, x0, x1, z0, z1, 'mm' if MM else 'm')
    return fig


def sube(m, h, paso=0.01):
    """(sol, atasco, terreno) de un escalon de altura h."""
    poli = terreno(h=h)
    _, s_de, _ = camino(h=h)
    S0 = 3.0 * s_de(np.array([-1.30, R]), poli)
    S1 = 3.0 * s_de(np.array([0.85, h + R]), poli)
    return barrido(m, poli, s_de, S0, S1, paso=paso) + (poli,)


def maximo_escalon(m, lo=0.0, hi=None, tol=0.002):
    """El escalon mas alto que esta geometria sube ENTERO, por biseccion.

    Merece la pena porque el resultado no es el que se esperaba: lo que corta
    la subida no es el tope del pivote -que se queda en 20 de 34 grados- sino
    que la llanta media se mete en la delantera.  Ese limite no estaba
    medido en ninguna parte."""
    hi = 2 * R if hi is None else hi
    if sube(m, hi)[1] is None:
        return hi
    while hi - lo > tol:
        med = 0.5 * (lo + hi)
        if sube(m, med)[1] is None:
            lo = med
        else:
            hi = med
    return lo


def holgura(m, sol):
    """Holgura entre llantas de un mismo lado, por estacion y por par.

    En reposo son 11.7 mm -el minimo que fijo el reescalado x0.862 de la
    huella- y el giro del balancin se los come, asi que hay que mirarla."""
    pares = (('front', 'mid'), ('mid', 'rear'), ('front', 'rear'))
    out = dict((p, []) for p in pares)
    for _, v, _, _ in sol:
        W, _, _ = ruedas(m, *v)
        for p in pares:
            out[p].append(math.hypot(*(W[p[0]] - W[p[1]])) - 2 * R)
    return dict((p, np.array(h)) for p, h in out.items())


def fase(q, h=ESCALON):
    """En que parte del terreno apoya una rueda, visto en su contacto."""
    if abs(q[0]) < 1e-9 and abs(q[1] - h) < 1e-9:
        return 'arista'
    if abs(q[1]) < 1e-9:
        return 'llano'
    if abs(q[0]) < 1e-9:
        return 'cara'
    return 'meseta'


def instantes(m, poli, sol, atasco=None):
    """Los fotogramas que cuentan la maniobra.

    Repartir el barrido por igual NO sirve: la subida se come una fraccion
    pequena del arco total y salian tres fotogramas en llano.  Se eligen por
    FASE DE CONTACTO: el llano, cada rueda mientras trepa por la cara o la
    arista, el instante en que la delantera ya pisa arriba y las otras dos no,
    y el final.  Cuantos salgan: si el vehiculo se atasca, las etapas que no
    llega a hacer no estan en el barrido y no se dibujan."""
    f = {}
    for i, (_, v, _, _) in enumerate(sol):
        W, _, _ = ruedas(m, *v)
        f[i] = dict((k, fase(contacto(W[k], poli)[1])) for k in W)

    def medio(cond):
        h = [i for i in sorted(f) if cond(f[i])]
        return h[len(h) // 2] if h else None

    trepa = ('cara', 'arista')
    fin = ('jammed' if atasco is not None else 'on the upper level')
    etapas = [
        (0, 'on the lower level'),
        (medio(lambda q: q['front'] in trepa), 'front wheel climbing'),
        (medio(lambda q: q['front'] == 'meseta' and q['mid'] == 'llano'),
         'front wheel up'),
        (medio(lambda q: q['mid'] in trepa), 'middle wheel climbing'),
        (medio(lambda q: q['rear'] in trepa), 'rear wheel climbing'),
        (len(sol) - 1, fin)]
    vistos, sel = set(), []
    for i, nom in etapas:
        if i is not None and i not in vistos:
            vistos.add(i)
            sel.append((i, nom))
    return sel


def fig_escalon(m, sol, atasco=None, h=ESCALON):
    """Los fotogramas, y SOLO los fotogramas.

    Fuera el grafico de angulos que iba debajo: sus tres curvas son datos,
    no un alzado, y en la misma figura que seis dibujos a linea competian
    por la atencion.  Los maximos van al pie de figura."""
    poli = terreno(h=h)
    terr = np.vstack([poli, [[3.0, -0.5], [-3.0, -0.5]]])
    sel = instantes(m, poli, sol, atasco)
    col = min(3, len(sel))
    fil = int(math.ceil(len(sel) / float(col)))
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 1.9 * fil + 0.3))
    gs = fig.add_gridspec(fil, col, hspace=0.26, wspace=0.05)

    for n, (k, nom) in enumerate(sel):
        ax = fig.add_subplot(gs[n // col, n % col])
        s, v, res, modo = sol[k]
        dibuja_robot(ax, m, v[0], v[1], v[2], v[3])
        ax.add_patch(Polygon(terr, closed=True, fc='none', hatch='/////',
                             ec='0.55', lw=0.0, zorder=0))
        ax.plot(poli[:, 0], poli[:, 1], '-', color='k', lw=1.0, zorder=1)
        ax.set_xlim(v[0] - 0.75, v[0] + 0.80)
        ax.set_ylim(-0.12, 1.00)
        ax.set_aspect('equal')
        # SIN EJES en los fotogramas: una escala de x por panel no decia nada
        # -cada uno sigue al vehiculo- y era la mitad de la tinta
        ax.set_axis_off()
        ax.set_title(u'(%s) %s' % ('abcdefgh'[n], nom), fontsize=7.5)
        ax.text(v[0] + 0.78, 0.98,
                u'pitch %+.1f$^\\circ$\nbogie %+.1f$^\\circ$'
                % (-math.degrees(v[2]), -math.degrees(v[3])),
                fontsize=7, ha='right', va='top', color='0.25')
        if n == 0:
            b = v[0] - 0.72
            ax.plot([b, b + 0.5], [0.90, 0.90], '-', color='k', lw=1.0)
            for x in (b, b + 0.5):
                ax.plot([x, x], [0.875, 0.925], '-', color='k', lw=1.0)
            ax.text(b + 0.25, 0.86, '0.5 m', fontsize=7, ha='center',
                    va='top', color='k')
        if n == 1:
            # la altura del escalon, una vez, y situada respecto al PANEL:
            # en coordenada absoluta se cortaba contra el panel vecino
            cota_v(ax, v[0] + 0.72, 0.0, h, cot(h), ref=0.0)
    return fig


# ------------------------------------------------------------------- main
def guarda(fig, out, nombre):
    for ext in ('eps', 'png'):
        fig.savefig(os.path.join(out, '%s.%s' % (nombre, ext)), dpi=200)
    plt.close(fig)
    print('  %s.eps / .png' % os.path.join(out, nombre))


def demo(m):
    """Comprobacion minima: el mecanismo en reposo y sobre el escalon."""
    assert abs(m['r'] - R) < 1e-9, 'el radio del xacro ya no es %.3f' % R
    W, piv, bp = ruedas(m, 0.0, R, 0.0, 0.0)
    assert max(abs(W[c][1] - R) for c in W) < 1e-5, \
        'las ruedas no son coplanarias en reposo'
    assert abs((W['front'][0] - W['mid'][0])
               - (W['mid'][0] - W['rear'][0])) < 1e-5, 'ejes no equidistantes'
    poli = terreno()
    assert abs(dist_frontera(np.array([-1.0, 0.5]), poli) - 0.5) < 1e-12
    assert abs(dist_frontera(np.array([1.0, 0.5]), poli) - 0.3) < 1e-12
    # arista: el punto a 45 deg de la esquina mide su propia distancia
    assert abs(dist_frontera(np.array([-0.1, ESCALON + 0.1]), poli)
               - math.hypot(0.1, 0.1)) < 1e-12
    # el camino arranca en el llano, acaba en la meseta, cada punto esta a un
    # radio del terreno, y s_de es la inversa de f
    f, s_de, largo = camino()
    for s in np.linspace(0.0, largo, 81):
        assert abs(dist_frontera(f(s), poli) - R) < 1e-9, \
            'el camino se sale del terreno en s=%.3f' % s
        assert abs(s_de(f(s), poli) - s) < 1e-9, \
            's_de no invierte a f en %.3f' % s
    assert abs(f(0.0)[1] - R) < 1e-12 and abs(f(largo)[1] - ESCALON - R) < 1e-9
    # llano: sin cabeceo y sin balancin, y los tres contactos cerrados
    S_ll = 3.0 * s_de(np.array([-1.2, R]), poli)
    for S in (S_ll, S_ll + 0.3, S_ll + 0.6):
        v = root(lambda u: residuos(m, poli, s_de, S, u),
                 np.array([-1.2, R, 0.0, 0.0])).x
        assert max(abs(np.array(residuos(m, poli, s_de, S, v)))) < 1e-9, \
            'el llano no resuelve en S=%.2f' % S
        # 1e-4 rad, no 0: el reescalado x0.862 dejo 1 um de dispersion en la
        # coplanaridad de las ruedas (ver rocker_huella_reescalada)
        assert abs(v[2]) < 1e-4 and abs(v[3]) < 1e-4, \
            'el llano sale cabeceado: %s' % v
    print('demo OK')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='.')
    p.add_argument('--rocker-drop', type=float, default=None)
    p.add_argument('--bogie-drop', type=float, default=None)
    p.add_argument("--paso", type=float, default=0.006)
    p.add_argument('--escalon', type=float, default=ESCALON,
                   help='altura del escalon en metros')
    p.add_argument('--metros', action='store_true',
                   help='acotar en metros en vez de en milimetros')
    p.add_argument('--demo', action='store_true')
    a = p.parse_args()

    global MM
    MM = not a.metros
    args = {}
    if a.rocker_drop is not None:
        args['rocker_drop'] = a.rocker_drop
    if a.bogie_drop is not None:
        args['bogie_drop'] = a.bogie_drop
    m = modelo(**args)
    if a.demo:
        return demo(m)

    sol, atasco, poli = sube(m, a.escalon, paso=a.paso)
    peor = max(r for _, _, r, _ in sol)
    print('residuo maximo de contacto  %.2e m' % peor)
    print('cabeceo  %+.1f .. %+.1f deg' % (
        -math.degrees(max(v[2] for _, v, _, _ in sol)),
        -math.degrees(min(v[2] for _, v, _, _ in sol))))
    print('bogie    %+.1f .. %+.1f deg  (tope %+.1f)' % (
        -math.degrees(max(v[3] for _, v, _, _ in sol)),
        -math.degrees(min(v[3] for _, v, _, _ in sol)),
        math.degrees(m['lim_bogie'][1])))
    for (a1, b1), h in sorted(holgura(m, sol).items()):
        print('holgura %s-%s  %+.1f .. %+.1f mm'
              % (a1, b1, 1000 * h.min(), 1000 * h.max()))
    if atasco is not None:
        print('ATASCO en x = %+.3f m: el escalon de %.0f mm NO se sube.  La '
              'rueda media llega a tocar la delantera y ya no puede ni subir '
              '-la llanta de la delantera se lo impide- ni avanzar -la cara '
              'del escalon se lo impide-.  Escalon maximo que sube entero: '
              '%.0f mm.' % (sol[-1][1][0], 1000 * a.escalon,
                            1000 * maximo_escalon(m)))
    mal = m.get('malla_rueda')
    if mal is None:
        print('AVISO: no se pudo medir la malla de la rueda; el plano acota '
              'la colision de todas formas, pero nadie comprueba que Gazebo '
              'dibuje esa misma rueda.')
    elif (max(abs(mal[[0, 2]] - R)) > 2e-4
          or abs(2 * mal[1] - m['ancho_llanta']) > 2e-4):
        print('AVISO: la MALLA de la rueda NO es la rueda que se simula.  El '
              'visual mide %.1f x %.1f mm de semieje (x, z) y %.1f de ancho, '
              'contra un cilindro de colision de %.1f de radio y %.1f de '
              'ancho.  En pantalla la rueda saldra de otro tamaño que en la '
              'fisica; con la malla pequeña no llega al chasis y el vehiculo '
              'parece flotar.  Escala que la cuadraria: %.8f %.6f %.8f, y hay '
              'que recalcular los seis <origin> del visual, porque el vertice '
              'cae en origin + escala * v.'
              % (1000 * mal[0], 1000 * mal[2], 2000 * mal[1], 1000 * R,
                 1000 * m['ancho_llanta'],
                 m['malla_escala'][0] * R / mal[0],
                 m['malla_escala'][1] * m['ancho_llanta'] / (2 * mal[1]),
                 m['malla_escala'][2] * R / mal[2]))
    else:
        print('malla de la rueda = colision: %.1f mm de radio y %.1f de '
              'ancho en los dos' % (1000 * mal[0], 2000 * mal[1]))
    print('holgura entre llantas segun el angulo del bogie, sin escalon de '
          'por medio:')
    for b in (-34.4, 0.0, 4.8, 20.0, 34.4):
        W, _, _ = ruedas(m, 0.0, R, 0.0, math.radians(-b))
        print('   bogie %+6.1f deg -> front-mid %+7.1f mm'
              % (b, 1000 * (math.hypot(*(W['front'] - W['mid'])) - 2 * R)))
    # EL VEREDICTO SALE DE LOS DATOS.  Estuvo escrito a mano -"el recorrido
    # declarado mete la llanta media en la delantera"- y al bajar la rueda a
    # 0.148 se quedo diciendo lo contrario de lo que median las cifras de
    # encima.  Ahora se decide con el peor angulo del recorrido.
    peor = min(math.hypot(*(ruedas(m, 0.0, R, 0.0, b)[0]['front']
                            - ruedas(m, 0.0, R, 0.0, b)[0]['mid'])) - 2 * R
               for b in np.linspace(m['lim_bogie'][0], m['lim_bogie'][1], 241))
    incl = math.degrees(math.atan2(-(m['mid'] - m['bogie'])[1],
                                   (m['mid'] - m['bogie'])[0]))
    alto = 1000 * (m['bogie'][1] - m['mid'][1])
    if peor < 0:
        print('O SEA: el recorrido declarado del bogie (%+.1f deg) METE la '
              'llanta media en la delantera, hasta %.1f mm.'
              % (math.degrees(m['lim_bogie'][1]), -1000 * peor))
    else:
        print('O SEA: en todo el recorrido del bogie (%+.1f deg) las llantas '
              'NO se tocan; lo mas justo son %.1f mm.'
              % (math.degrees(m['lim_bogie'][1]), 1000 * peor))
    print('      El brazo del balancin al eje medio esta inclinado %.1f deg '
          '-el pivote va %.0f mm por encima de los ejes- asi que girarlo para '
          'SUBIR la media la echa hacia DELANTE; con r = %.3f eso ya cabe.'
          % (incl, alto, R))
    print('      escalon maximo que sube entero: %.0f mm'
          % (1000 * maximo_escalon(m)))

    out = os.path.expanduser(a.out)
    guarda(fig_dimensiones(m), out, 'rocker_2d_dimensiones')
    guarda(fig_escalon(m, sol, atasco, a.escalon), out, 'rocker_2d_escalon')
    pies(m, sol, atasco, a.escalon)


def pies(m, sol, atasco=None, esc=ESCALON):
    """Pies de figura, en ingles y listos para pegar.

    Los planos no llevan titulo ni leyenda, asi que todo lo que no es una
    medida -que pivote es cual, sus topes, masa, huella, K- vive aqui, que es
    donde un paper lo pone."""
    u = 'mm' if MM else 'm'
    be = max(abs(math.degrees(v[3])) for _, v, _, _ in sol)
    th = max(abs(math.degrees(v[2])) for _, v, _, _ in sol)
    h = holgura(m, sol)['front', 'mid']
    print('\n--- captions (English, paste as-is) ---')
    k = 1000.0 if MM else 1.0
    print('Fig. 1  Rocker-bogie suspension, side elevation, dimensions in %s, '
          'measured from the wheel contact plane. The pivot above the middle '
          'wheel is the rocker pivot (travel +-%.1f rad) and the one behind '
          'it the bogie pivot (+-%.1f rad); both lie at the same height. '
          'Wheel radius is the collision cylinder, which is what the '
          'simulation uses. Not visible in this plane: front track %.1f, '
          'middle and rear track %.1f, chassis width %.1f, wheel width %.1f, '
          'Velodyne ray height %.1f, camera %.1f, IMU %.1f. Total mass '
          '%.3f kg; support-polygon area %.4f m2; pivot-to-contact lever '
          'K = %.2f.'
          % (u, m['lim_rocker'][1], m['lim_bogie'][1],
             k * m['via_del'], k * m['via_tras'], k * m['chasis_ancho'],
             k * m['ancho_llanta'], k * (m['lidar_rayo'] + R),
             k * (m['camera_link'][1] + R), k * (m['imu_link'][1] + R),
             m['masa'], m['huella'], (m['rocker'][1] + R) / R))
    com = ('Quasi-static attempt at a %.0f %s step' if atasco is not None
           else 'Quasi-static climb of a %.0f %s step')
    pie = (com + ', taller than the wheel radius, with the rocker held by an '
           'ideal differential (a step across the full width pitches both '
           'rockers equally, so the rocker is rigid with respect to the '
           'chassis and only the bogie articulates). Wheel-to-wheel contact '
           'is enforced, so no pose interpenetrates; the tightest gap between '
           'the front and middle tyres is %.1f mm. Peak chassis pitch %.1f '
           'deg, peak bogie angle %.1f deg against a %.1f deg stop.'
           ) % (1000 * esc if MM else esc, u, 1000 * h.min(), th, be,
                math.degrees(m['lim_bogie'][1]))
    if atasco is not None:
        pie += (' The climb is not completed: the joint travel is never the '
                'limit, wheel-to-wheel interference is. The bogie arm to the '
                'middle axle is inclined %.1f deg, so rotating it to raise '
                'that wheel also swings it forward, and the gap to the front '
                'tyre is only %.1f mm at rest. The vehicle wedges with the '
                'middle wheel unable either to rise, blocked by the front '
                'tyre, or to advance, blocked by the step face. The tallest '
                'step this geometry clears is %.0f %s.'
                % (math.degrees(math.atan2(-(m['mid'] - m['bogie'])[1],
                                           (m['mid'] - m['bogie'])[0])),
                   1000 * h[0],
                   1000 * maximo_escalon(m) if MM else maximo_escalon(m), u))
    print('Fig. 2  ' + pie)


if __name__ == '__main__':
    main()
