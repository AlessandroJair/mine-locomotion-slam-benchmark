#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banco cuasi-estatico de la suspension del rocker-bogie, SIN Gazebo.

PARA QUE.  Cada tanteo de geometria costaba una campana de 5 minutos de reloj
por corrida mas el analisis, y la mayoria de los tanteos se descartan.  Esto
contesta en segundos la unica pregunta que ha decidido todos los cambios hasta
ahora: CUANTO TIEMPO RUEDA CON MENOS DE SEIS RUEDAS EN EL SUELO, y cual.

QUE MODELA Y QUE NO.  Es cuasi-estatico: resuelve, en cada estacion de la ruta,
la configuracion de minima energia potencial compatible con que ninguna rueda
penetre el terreno.  Para una suspension PASIVA sin muelles eso ES el
equilibrio.  Lo que NO tiene:

  * dinamica.  Ni inercia, ni rebote de contacto, ni el castaneteo que domina
    la vibracion RMS.  No prediga nadie vibracion con esto.
  * par de rueda.  La nota de bogie_drop demuestra que el par enrolla el
    balancin con palanca K = h_huella/r; aqui no esta.  O sea que el modelo
    SUBESTIMA la perdida de contacto, y lo hace mas cuanto mayor sea K.
  * direccion.  Se supone recta; el angulo de las manguetas no se registra.

LO QUE CONTESTA, Y NO ES LO QUE SE BUSCABA (--validar, terreno de drop4):

                    predicho n<6     medido n<6
      drop3             0.13 %         17.99 %
      drop4             0.00 %         13.16 %
      drop5             0.13 %         32.66 %

Las tres geometrias se adaptan al terreno PERFECTAMENTE: el mecanismo llega al
suelo con las seis ruedas en el 99.9 % de las estaciones, y las tres por igual.
Eso no es que el modelo falle, es un RESULTADO, y es fuerte:

    NADA de la perdida de contacto medida (13-33 %) es cinematico.

No es que a la suspension le falte recorrido ni que la geometria no de. Es
fuerza: el par de rueda enrollando los pivotes pasivos contra su tope, que es
justo lo que la nota de bogie_drop del xacro demostro con K = h_huella/r y lo
que la de bogie_upper describio -"el tope no ataba, SOSTENIA"-. Los pivotes no
llevan muelle, solo damping y friccion, que resisten pero no devuelven, asi que
un par constante los lleva hasta donde le dejen.

CONSECUENCIA PRACTICA: este banco NO ordena geometrias -lo dice el mismo si se
le pide --barrido- y no sirve para ahorrarse la campana si lo que se busca es
el porcentaje de contacto. Para lo que SI sirve, y es lo que hace falta ahora:

  * SEPARAR lo geometrico de lo que no lo es.  drop5 sale igual de conforme
    que drop4 y medida es 2.5 veces peor: la regresion de drop5 no esta en la
    geometria de la suspension.
  * comprobar que un cambio no rompe la conformidad antes de lanzar nada
    (un rocker_drop que pase de su techo saldria aqui en segundos).
  * el modo diferencial: |phi_l - phi_r| sale 2.11-2.18 deg contra 1.36 / 1.36
    / 0.83 medidos, o sea el orden de magnitud bueno.

PARA QUE PREDIGA EL PORCENTAJE hay que meter las FUERZAS: repartir la carga por
el varillaje -momento de cada balancin sobre su pivote, momento de cada rocker,
y el diferencial acoplando los dos- y declarar que una rueda se levanta cuando
su normal se anula. Con eso entra el par de rueda y con el la palanca K, que es
la variable que todos estos cambios han estado moviendo. Es la continuacion
natural y no esta hecha.

EL TERRENO SALE DE LAS PROPIAS CORRIDAS.  No hace falta leer la malla de la
mina: una corrida ya medida lleva, en cada instante, la pose del chasis y los
cuatro angulos de la suspension, asi que se puede reconstruir donde estaba el
suelo bajo cada rueda que tocaba.  Y como el reescalado dejo la geometria en X
y en Y CONGELADA, las seis ruedas de cualquier variante pisan exactamente la
misma huella: el perfil se puede reutilizar tal cual.  Se construye un perfil
z(s) por oruga -izquierda y derecha- indexado por arco recorrido.

    python3 suspension_sim.py --validar
    python3 suspension_sim.py --barrido
    python3 suspension_sim.py --rocker-drop 0.17 --bogie-drop 0.05
"""
from __future__ import print_function

import argparse
import csv
import io
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import root

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
XACRO = os.path.join(RAIZ, 'src', 'rocker_bogie', 'urdf', 'ensamblajeurdf.xacro')

R_RUEDA = 0.148
# izquierda y derecha en el convenio de base_link: LEFT = y menor
RUEDAS = {'Rueda_6_1': ('L', 'front'), 'Rueda_4_1': ('L', 'mid'),
          'Rueda_2_1': ('L', 'rear'),
          'Rueda_1_1': ('R', 'front'), 'Rueda_3_1': ('R', 'mid'),
          'Rueda_5_1': ('R', 'rear')}
PIV_LADO = {'L': 'rocker_pivot_left', 'R': 'rocker_pivot_right'}
BOG_LADO = {'L': 'rev_3', 'R': 'rev_4'}


# ---------------------------------------------------------------- geometria
def genera_urdf(**args):
    """El xacro lleva $(find rocker_bogie), asi que xacro necesita el
    workspace en ROS_PACKAGE_PATH.  Sin `source devel/setup.bash` el unico
    sintoma era `CalledProcessError: returned non-zero exit status 2`, porque
    el stderr con el "resource not found: rocker_bogie" se iba al PIPE y
    nadie lo leia.  Se anade src/ al PATH de paquetes -no hace falta nada
    mas del workspace- y si xacro falla de todas formas, se ensena su
    stderr."""
    cmd = ['xacro', XACRO, 'differential:=false', 'differential_linkage:=true',
           'bogie_upper:=0.6']
    for k, v in args.items():
        cmd.append('%s:=%s' % (k, v))
    env = dict(os.environ)
    src = os.path.join(RAIZ, 'src')
    if src not in env.get('ROS_PACKAGE_PATH', '').split(':'):
        env['ROS_PACKAGE_PATH'] = ':'.join(
            [src] + [p for p in [env.get('ROS_PACKAGE_PATH')] if p])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env)
    if p.returncode != 0:
        raise RuntimeError('xacro fallo (%d):\n%s'
                           % (p.returncode, p.stderr.decode()))
    return p.stdout.decode()


def geometria(**args):
    """El varillaje de un lado, reducido a su plano x-z.

    Todas las juntas pasivas giran sobre +-Y, asi que cada lado es un
    mecanismo PLANO: pivote del rocker, brazo al delantero, brazo al pivote
    del bogie, y del bogie dos brazos a la media y a la trasera.  Los dos
    lados solo se hablan a traves del chasis (y del varillaje, si esta)."""
    root = ET.fromstring(genera_urdf(**args))
    J, lim = {}, {}
    for j in root.findall('joint'):
        o = j.find('origin')
        xyz = np.array([float(v) for v in (o.get('xyz') or '0 0 0').split()])
        J[j.find('child').get('link')] = (j.find('parent').get('link'), xyz)
        l = j.find('limit')
        if l is not None and l.get('lower') is not None:
            lim[j.get('name')] = (float(l.get('lower')), float(l.get('upper')))
    masas = {}
    for ln in root.findall('link'):
        i = ln.find('inertial')
        if i is None:
            continue
        o = i.find('origin')
        c = np.array([float(v) for v in ((o.get('xyz') if o is not None
                                          else None) or '0 0 0').split()])
        masas[ln.get('name')] = (float(i.find('mass').get('value')), c)

    def pos(link, hasta='chassis_link'):
        p = np.zeros(3)
        while link != hasta and link in J:
            par, d = J[link]
            p = p + d
            link = par
        return p

    g = {'lim_rocker': lim['rocker_pivot_left'], 'lim_bogie': lim['rev_3'],
         'masa_total': sum(m for m, _ in masas.values())}
    # CM del chasis y de todo lo que va rigido con el
    rig = [n for n in masas if n not in J or _cuelga_de_pivote(n, J) is False]
    g['cm_chasis'] = _cm(masas, J, pos, rig)
    g['m_chasis'] = sum(masas[n][0] for n in rig)
    # LEFT = y menor, que es el convenio de base_link (mira a -x)
    tubos = sorted([c for c, (p, _) in J.items()
                    if p == 'chassis_link' and c.startswith('AcopleTuboRB')],
                   key=lambda c: J[c][1][1])
    soportes = sorted([c for c in J if c.startswith('Acople_Rodamiento')],
                      key=lambda c: pos(c)[1])
    for lado, tubo, sop in (('L', tubos[0], soportes[0]),
                            ('R', tubos[1], soportes[1])):
        piv = J[tubo][1]
        g[lado + '_pivote'] = piv
        g[lado + '_bogie'] = pos(sop) - piv
        for w, (ld, cual) in RUEDAS.items():
            if ld != lado:
                continue
            if cual == 'front':
                g[lado + '_front'] = pos(w) - piv
            else:
                g['%s_%s_b' % (lado, cual)] = pos(w) - pos(sop)
    return g


def _cuelga_de_pivote(link, J):
    while link in J:
        par, _ = J[link]
        if par == 'chassis_link':
            return link.startswith('AcopleTuboRB')
        link = par
    return False


def _cm(masas, J, pos, nombres):
    M = sum(masas[n][0] for n in nombres)
    S = np.zeros(3)
    for n in nombres:
        m, c = masas[n]
        S += m * (pos(n) + c)
    return S / M


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ruedas_chasis(g, phiL, phiR, betL, betR):
    """Centro de las seis ruedas en el marco del chasis."""
    out = {}
    for lado, phi, bet in (('L', phiL, betL), ('R', phiR, betR)):
        P, Rp = g[lado + '_pivote'], _rot_y(phi)
        out[lado + 'front'] = P + Rp.dot(g[lado + '_front'])
        B = P + Rp.dot(g[lado + '_bogie'])
        Rb = _rot_y(phi + bet)
        for cual in ('mid', 'rear'):
            out[lado + cual] = B + Rb.dot(g['%s_%s_b' % (lado, cual)])
    return out


# ------------------------------------------------------------------ terreno
def lee_run(tag):
    p = ('/home/alessandro/metrics_rocker_bogie_%s/fixed_trajectory/'
         'swept_lidar/rocker_bogie/run01/metrics.csv' % tag)
    with io.open(p, newline='') as f:
        return list(csv.DictReader(f))


def perfiles(rows, g, paso=0.02):
    """z(s) de la oruga izquierda y de la derecha, reconstruido de la corrida.

    Para cada instante y cada rueda que TOCA, el suelo esta un radio por
    debajo de su centro.  Se acumula por arco recorrido y se promedia."""
    acc = {'L': {}, 'R': {}}
    for r in rows:
        s = float(r['gt_distance[m]'])
        ro, pi = float(r['roll[rad]']), float(r['pitch[rad]'])
        zc = float(r['gt_z[m]'])
        W = ruedas_chasis(g,
                          float(r['suspension_rocker_pivot_left[rad]']),
                          float(r['suspension_rocker_pivot_right[rad]']),
                          float(r['suspension_rev_3[rad]']),
                          float(r['suspension_rev_4[rad]']))
        Rm = _rot_rp(ro, pi)
        for w, (lado, cual) in RUEDAS.items():
            if float(r['contact_%s[-]' % w]) < 0.5:
                continue
            p = Rm.dot(W[lado + cual])
            # arco de ESA rueda.  base_link mira a -x, asi que el adelanto
            # longitudinal en arco es -x, no +x.
            k = int(round((s - p[0]) / paso))
            acc[lado].setdefault(k, []).append(zc + p[2] - R_RUEDA)
    out = {}
    for lado in 'LR':
        ks = sorted(acc[lado])
        out[lado] = (np.array(ks) * paso,
                     np.array([np.median(acc[lado][k]) for k in ks]))
    return out


def _rot_rp(roll, pitch):
    cr, sr, cp, sp = (math.cos(roll), math.sin(roll),
                      math.cos(pitch), math.sin(pitch))
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry.dot(Rx)


def suelo(perf, lado, s):
    x, z = perf[lado]
    return float(np.interp(s, x, z))


# --------------------------------------------------------------- equilibrio
# POR QUE UN SISTEMA DE ECUACIONES Y NO UN MINIMO DE ENERGIA.  La primera
# version minimizaba la altura del centro de masas sujeta a que ninguna rueda
# penetrase el terreno.  Suena bien y esta mal: con DESIGUALDADES el mecanismo
# queda suelto -una rueda en el aire no restringe nada- y el optimizador abria
# los dos rockers a topes opuestos para bajar el chasis.  Convergia el 28 % de
# las veces y sacaba |phi_l - phi_r| de 19.6 deg de media contra los 1.36 deg
# medidos en Gazebo.
#
# Con las seis ruedas TOCANDO, el sistema esta exactamente determinado:
#     7 incognitas  zc, balanceo, cabeceo, phi_l, phi_r, beta_l, beta_r
#     7 ecuaciones  6 huelgos = 0, mas el diferencial (phi_l + phi_r = 0)
# asi que no hay familia de soluciones ni nada que "elegir": se resuelve y ya.
#
# QUE SIGNIFICA ENTONCES PERDER CONTACTO.  Si la solucion pide un pivote FUERA
# de su tope, el mecanismo no llega al suelo con las seis: esa estacion se
# cuenta como perdida de contacto, y la rueda que se levanta es la del extremo
# del brazo saturado.  Es un criterio CINEMATICO -conformidad del mecanismo-,
# no de fuerzas: no sabe de par de rueda ni de dinamica, asi que subestima.
MAX_IT = 40


def equilibrio(g, perf, s, x0, dif='ideal', tol=None, lim_dif=None):
    lo_r, hi_r = g['lim_rocker']
    lo_b, hi_b = g['lim_bogie']

    def huelgos(v):
        zc, ro, pi, pL, pR, bL, bR = v
        Rm = _rot_rp(ro, pi)
        W = ruedas_chasis(g, pL, pR, bL, bR)
        h = []
        for w, (lado, cual) in sorted(RUEDAS.items()):
            p = Rm.dot(W[lado + cual])
            h.append((zc + p[2] - R_RUEDA) - suelo(perf, lado, s - p[0]))
        return np.array(h)

    def F(v):
        h = huelgos(v)
        if dif == 'ideal':
            return np.append(h, v[3] + v[4])
        # sin varillaje el mecanismo tiene un grado de libertad de sobra: se
        # cierra pidiendo que el chasis no balancee mas de lo que pide el suelo
        return np.append(h, 0.0)

    # DOS ARRANQUES, Y SE QUEDA EL MAS CERCANO A NEUTRO.  El sistema tiene
    # mas de una raiz: con arranque en caliente el solver se metia en una rama
    # con los rockers abiertos +-11 deg -converge, residuo bueno, y fisicamente
    # imposible con este terreno- y de ahi no salia en toda la corrida, lo que
    # metia un 37 % de falsa perdida de contacto en una sola fila del barrido.
    # Con un terreno suave la rama buena es la de menor articulacion.
    mejor = None
    for arranque in (x0, np.array([x0[0], 0, 0, 0, 0, 0, 0])):
        sl = root(F, arranque, method='hybr', options={'maxfev': MAX_IT * 8})
        if float(np.abs(F(sl.x)).max()) >= 0.005:
            continue
        pena = float(np.abs(sl.x[3:]).sum())
        if mejor is None or pena < mejor[0]:
            mejor = (pena, sl)
    sol = mejor[1] if mejor else root(F, x0, method='hybr',
                                      options={'maxfev': MAX_IT * 8})
    v = sol.x
    # CONVERGENCIA APARTE.  Un residuo grande significa que no se sabe donde
    # esta el equilibrio, no que haya una rueda levantada.  Contar esas
    # estaciones como perdida de contacto metia picos de 37 % en el barrido a
    # partir de una sola divergencia.  Se descartan y se informa de cuantas.
    conv = float(np.abs(F(v)).max()) < 0.005
    # saturacion: el mecanismo no llega
    sat = np.array([v[3] < lo_r, v[3] > hi_r, v[4] < lo_r, v[4] > hi_r,
                    v[5] < lo_b, v[5] > hi_b, v[6] < lo_b, v[6] > hi_b])
    if lim_dif is not None and abs(math.degrees(v[3] - v[4])) > lim_dif:
        sat = np.append(sat, True)
    cont = np.ones(6, dtype=bool)
    if sat.any() or not sol.success:
        # la rueda que se va al aire es la del brazo que se paso de tope
        for k, w in enumerate(sorted(RUEDAS)):
            lado, cual = RUEDAS[w]
            idx_r = 3 if lado == 'L' else 4
            idx_b = 5 if lado == 'L' else 6
            fuera_r = not (lo_r - 1e-9 <= v[idx_r] <= hi_r + 1e-9)
            fuera_b = not (lo_b - 1e-9 <= v[idx_b] <= hi_b + 1e-9)
            if cual == 'front' and fuera_r:
                cont[k] = False
            if cual in ('mid', 'rear') and fuera_b:
                cont[k] = False
        if lim_dif is not None and abs(math.degrees(v[3] - v[4])) > lim_dif:
            # el varillaje impide el balanceo relativo: se queda sin apoyo la
            # delantera del lado que tendria que bajar mas
            k = sorted(RUEDAS).index('Rueda_6_1' if v[3] > v[4] else 'Rueda_1_1')
            cont[k] = False
        if not cont.all() is False and not sol.success:
            cont[:] = cont & True
    v = np.clip(v, [-5, -1.5, -1.5, lo_r, lo_r, lo_b, lo_b],
                [5, 1.5, 1.5, hi_r, hi_r, hi_b, hi_b])
    return v, cont, conv


def replay(g, perf, rows, dif='ideal', cada=10, lim_dif=None):
    """Recorre la ruta y cuenta contacto."""
    x0 = np.array([0.2, 0, 0, 0, 0, 0, 0])
    n_tot = n_menos6 = n_menos5 = n_malo = 0
    por_rueda = dict((w, 0) for w in sorted(RUEDAS))
    cab, dm = [], []
    for i in range(0, len(rows), cada):
        s = float(rows[i]['gt_distance[m]'])
        x0, cont, conv = equilibrio(g, perf, s, x0, dif, lim_dif=lim_dif)
        if not conv:
            n_malo += 1
            continue
        n_tot += 1
        c = int(cont.sum())
        n_menos6 += (c < 6)
        n_menos5 += (c <= 4)
        for k, w in enumerate(sorted(RUEDAS)):
            por_rueda[w] += bool(cont[k])
        cab.append(abs(math.degrees(x0[2])))
        dm.append(abs(math.degrees(x0[3] - x0[4])))
    cab.sort()
    if not n_tot:
        return {'n<6': float('nan'), 'n<=4': float('nan'),
                'cabeceo p95': float('nan'), 'modo dif medio': float('nan'),
                'descartadas': 100.0, 'contacto': {}}
    return {'descartadas': 100.0 * n_malo / (n_tot + n_malo), 'n<6': 100.0 * n_menos6 / n_tot, 'n<=4': 100.0 * n_menos5 / n_tot,
            'cabeceo p95': cab[int(0.95 * (len(cab) - 1))],
            'modo dif medio': sum(dm) / len(dm),
            'contacto': dict((w, 100.0 * v / n_tot)
                             for w, v in por_rueda.items())}


# ------------------------------------------------------------------- salida
# Lo medido en Gazebo, para que la validacion no dependa de volver a leerlo.
MEDIDO = {
    'drop3': dict(rocker_drop=0.0, bogie_drop=0.0,
                  **{'n<6': 17.99, 'n<=4': 4.67, 'cabeceo max': 28.40}),
    'drop4': dict(rocker_drop=0.170395, bogie_drop=0.0,
                  **{'n<6': 13.16, 'n<=4': 2.73, 'cabeceo max': 29.08}),
    'drop5': dict(rocker_drop=0.299477, bogie_drop=0.0,
                  **{'n<6': 32.66, 'n<=4': 5.10, 'cabeceo max': 28.58}),
}
REF = 'drop4'     # de donde sale el terreno: la corrida mas limpia de las tres


def _fila(nom, p, med=None):
    e = ''
    if med is not None:
        e = '   medido %5.2f / %4.2f   error %+5.2f' % (
            med['n<6'], med['n<=4'], p['n<6'] - med['n<6'])
    print('%-22s n<6 %5.2f %%   n<=4 %4.2f %%   |dif| %4.2f deg   '
          'sin converger %4.1f %%%s'
          % (nom, p['n<6'], p['n<=4'], p['modo dif medio'],
             p['descartadas'], e))


def main():
    ap = argparse.ArgumentParser(
        description='Banco cuasi-estatico de la suspension, sin Gazebo.')
    ap.add_argument('--validar', action='store_true',
                    help='reproduce drop3/4/5 y compara con lo medido')
    ap.add_argument('--barrido', action='store_true',
                    help='barre rocker_drop x bogie_drop y ordena')
    ap.add_argument('--rocker-drop', type=float, default=None)
    ap.add_argument('--bogie-drop', type=float, default=0.0)
    ap.add_argument('--dif', default='ideal', choices=['ideal', 'libre'],
                    help='varillaje del diferencial: ata el modo comun o no')
    ap.add_argument('--limite-dif', type=float, default=None,
                    help='tope duro al modo diferencial |phi_l-phi_r|, en '
                         'grados.  Sin el, el modo esta libre.')
    ap.add_argument('--ajusta-limite', action='store_true',
                    help='busca que tope del modo diferencial reproduce el '
                         'n<6 medido de drop5')
    ap.add_argument('--cada', type=int, default=20,
                    help='resuelve una de cada N muestras (50 Hz)')
    a = ap.parse_args()

    rows = lee_run(REF)
    g_ref = geometria(rocker_drop=MEDIDO[REF]['rocker_drop'],
                      bogie_drop=MEDIDO[REF]['bogie_drop'])
    perf = perfiles(rows, g_ref)
    print('terreno reconstruido de %s: %d muestras izq, %d der, %.1f m\n'
          % (REF, len(perf['L'][0]), len(perf['R'][0]), max(perf['L'][0])))

    if a.ajusta_limite:
        m = MEDIDO['drop5']
        g = geometria(rocker_drop=m['rocker_drop'], bogie_drop=m['bogie_drop'])
        print('drop5 medido: n<6 %.2f %%, |phi_l-phi_r| medio 0.83 deg\n'
              % m['n<6'])
        for lim in (None, 8.0, 4.0, 2.0, 1.2, 0.8, 0.5):
            p = replay(g, perf, rows, a.dif, a.cada, lim_dif=lim)
            _fila('tope %s' % ('libre' if lim is None else '%.1f deg' % lim), p)
        print('\nEl tope que reproduce el 32.66 %% medido es la rigidez que el'
              '\nvarillaje le esta metiendo al modo que deberia dejar suelto.')
        return

    if a.validar:
        print('VALIDACION - el terreno sale de %s, asi que esa fila es la que'
              ' mas tiene que acertar' % REF)
        for tag, m in sorted(MEDIDO.items()):
            g = geometria(rocker_drop=m['rocker_drop'],
                          bogie_drop=m['bogie_drop'])
            _fila(tag, replay(g, perf, rows, a.dif, a.cada,
                              lim_dif=a.limite_dif), m)
        print('\nSi drop5 sale MEJOR que drop4 y lo medido dice lo contrario,'
              '\nel modelo esta diciendo que la regresion NO es geometrica.')
        return

    if a.barrido:
        print('%-22s %s' % ('rocker / bogie', 'predicho'))
        res = []
        for rd in (0.0, 0.085, 0.170395, 0.24, 0.299477, 0.34):
            tope = 0.351444 - max(0.0, rd - 0.170395)
            for bd in (0.0, min(0.09, tope), min(0.178, tope)):
                g = geometria(rocker_drop=rd, bogie_drop=bd)
                p = replay(g, perf, rows, a.dif, a.cada,
                           lim_dif=a.limite_dif)
                res.append((p['n<6'], rd, bd, p))
                _fila('%.3f / %.3f' % (rd, bd), p)
        spread = max(r[0] for r in res) - min(r[0] for r in res)
        # el umbral es RELATIVO a la resolucion: con N estaciones, una sola que
        # cambie ya mueve 100/N puntos, asi que por debajo de cinco estaciones
        # de diferencia no hay nada que ordenar
        res_pt = 100.0 / max(1, len(range(0, len(rows), a.cada)))
        if spread < 5 * res_pt:
            print('\nNO SE PUEDE ORDENAR: entre la mejor y la peor hay %.2f'
                  ' puntos y la resolucion es %.2f: es ruido.'
                  % (spread, res_pt))
            print('El criterio de este banco es CINEMATICO -la rueda se levanta'
                  ' cuando el\nmecanismo no llega al suelo sin pasarse de'
                  ' tope- y en este terreno eso casi\nno pasa en ninguna'
                  ' geometria.  Lo que separa a unas de otras en Gazebo es el'
                  '\nPAR DE RUEDA enrollando el balancin contra su tope, y eso'
                  ' aqui no esta.')
            print('\nMientras no se meta, usar --validar (que si separa lo'
                  ' geometrico de lo que\nno lo es) y no este barrido.')
            return
        print('\nMEJORES:')
        for v, rd, bd, p in sorted(res)[:3]:
            print('  rocker_drop %.6f  bogie_drop %.3f  -> n<6 %.2f %%'
                  % (rd, bd, v))
        return

    rd = a.rocker_drop
    if rd is None:
        rd = MEDIDO['drop5']['rocker_drop']
    g = geometria(rocker_drop=rd, bogie_drop=a.bogie_drop)
    p = replay(g, perf, rows, a.dif, a.cada, lim_dif=a.limite_dif)
    _fila('rocker %.4f bogie %.3f' % (rd, a.bogie_drop), p)
    for w in sorted(p['contacto']):
        print('    %-10s %5.2f %%' % (w, p['contacto'][w]))


if __name__ == '__main__':
    main()
