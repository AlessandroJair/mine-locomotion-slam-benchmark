#!/usr/bin/env python3
"""Shared schema and I/O for the robot_metrics CSV files.

Every column carries its unit in the header, as "name[unit]", so a CSV is
self-describing and the units propagate automatically into the generated
tables (and from there into Table 3 of the paper) instead of living only in
someone's memory.

Dimensionless counts and flags use "[-]".

Both the logger and the analysis scripts import this module, so the schema
cannot drift between the code that writes the data and the code that reads it.

    from metrics_io import METRIC_COLUMNS, csv_header, load_metrics

    load_metrics(path) returns a DataFrame whose column names have the unit
    suffix stripped ("pitch" not "pitch[rad]"), with the units available as
    df.attrs['units'].  Legacy files written before the unit convention are
    read transparently.
"""

import csv
import os
import sys

# ---------------------------------------------------------------------------
# (name, unit, description)
# ---------------------------------------------------------------------------
METRIC_COLUMNS = [
    ('timestamp', 's', 'ROS sim time (use_sim_time:=true)'),

    # --- attitude / IMU ---
    ('pitch', 'rad', 'Chassis pitch from IMU orientation'),
    ('roll', 'rad', 'Chassis roll from IMU orientation'),
    ('yaw', 'rad', 'Chassis yaw from IMU orientation'),
    ('accel_x', 'm/s^2', 'IMU linear acceleration, body x'),
    ('accel_y', 'm/s^2', 'IMU linear acceleration, body y'),
    ('accel_z', 'm/s^2', 'IMU linear acceleration, body z (includes g)'),
    ('angular_vel_x', 'rad/s', 'IMU angular velocity, body x'),
    ('angular_vel_y', 'rad/s', 'IMU angular velocity, body y'),
    ('angular_vel_z', 'rad/s', 'IMU angular velocity, body z'),
    ('angular_accel_x', 'rad/s^2', 'd(angular_vel_x)/dt, numerical'),
    ('angular_accel_z', 'rad/s^2', 'd(angular_vel_z)/dt, numerical'),

    # --- SLAM odometry estimate ---
    ('odom_x', 'm', 'SLAM position estimate, x'),
    ('odom_y', 'm', 'SLAM position estimate, y'),
    ('odom_z', 'm', 'SLAM position estimate, z'),
    ('odom_roll', 'rad', 'SLAM orientation estimate, roll'),
    ('odom_pitch', 'rad', 'SLAM orientation estimate, pitch'),
    ('odom_yaw', 'rad', 'SLAM orientation estimate, yaw'),
    ('odom_vx', 'm/s', 'SLAM linear velocity estimate, body x'),
    ('odom_vyaw', 'rad/s', 'SLAM yaw rate estimate'),

    # --- command ---
    ('cmd_vx', 'm/s', 'Commanded linear velocity'),
    ('cmd_vyaw', 'rad/s', 'Commanded yaw rate'),

    # --- ground truth (exact Gazebo model pose, unfiltered) ---
    ('gt_x', 'm', 'Gazebo ground-truth position, x (world frame)'),
    ('gt_y', 'm', 'Gazebo ground-truth position, y (world frame)'),
    ('gt_z', 'm', 'Gazebo ground-truth position, z (world frame)'),
    ('gt_roll', 'rad', 'Gazebo ground-truth roll'),
    ('gt_pitch', 'rad', 'Gazebo ground-truth pitch'),
    ('gt_yaw', 'rad', 'Gazebo ground-truth yaw'),
    ('gt_vx', 'm/s', 'Gazebo ground-truth linear velocity, world x'),
    ('gt_vy', 'm/s', 'Gazebo ground-truth linear velocity, world y'),
    # Proper (kinematic) acceleration from the ground-truth twist, rotated
    # into the body frame so it is directly comparable to accel_* above.
    # It carries NO gravity term and no sensor noise, which is why the
    # vibration metrics read these and not the IMU: see the VIBRATION note
    # in slam_metrics.evaluate_stability_correlation().
    ('gt_ax', 'm/s^2', 'Ground-truth proper acceleration, body x (no gravity)'),
    ('gt_ay', 'm/s^2', 'Ground-truth proper acceleration, body y (no gravity)'),
    ('gt_az', 'm/s^2', 'Ground-truth proper acceleration, body z (no gravity)'),
    ('gt_distance', 'm', 'Cumulative ground-truth path length'),

    # --- SLAM internals ---
    ('loop_closure_count', '-', 'Cumulative loop closures accepted'),
    ('proximity_detection_count', '-', 'Cumulative proximity detections'),
    ('slam_inliers', '-', 'Visual/ICP inliers of the last odometry update'),
    ('slam_matches', '-', 'Feature matches of the last odometry update'),
    ('slam_icp_inliers_ratio', '-', 'ICP inlier ratio'),
    ('slam_icp_correspondences', '-', 'ICP correspondences'),
    ('slam_processing_time', 's', 'Odometry estimation time'),
    ('tracking_lost', '-', '1 while odometry tracking is lost, else 0'),
    ('tracking_loss_count', '-', 'Cumulative tracking-loss events'),
    ('relocalization_count', '-', 'Cumulative recoveries after a loss'),

    # --- map->odom correction ---
    ('tf_correction_magnitude', 'm', 'Per-sample |delta| of the map->odom TF'),
    ('tf_correction_yaw', 'rad', 'Per-sample yaw delta of the map->odom TF'),
]

# Per-wheel contact columns are appended dynamically: one contact flag and one
# normal force per wheel link.
CONTACT_COLUMN_TEMPLATES = [
    ('contact_{}', '-', 'Wheel {} in contact with the ground (1/0)'),
    ('normal_force_{}', 'N', 'Wheel {} contact normal force magnitude'),
]


def contact_columns(wheel_links):
    cols = []
    for link in wheel_links:
        for name, unit, desc in CONTACT_COLUMN_TEMPLATES:
            cols.append((name.format(link), unit, desc.format(link)))
    return cols


def suspension_columns(joints):
    """Una columna por articulacion pasiva de la suspension, en radianes.

    Existen porque sin ellas no se puede distinguir "la suspension articula y
    el terreno le gana" de "la suspension esta contra sus topes".  Medido el
    2026-09-10 sobre metrics_lo_que_sea: el rocker_bogie pasa el 15.2 % del
    tiempo con menos de sus seis ruedas en el suelo, el 69 % de ese tiempo en
    el 20 % mas agitado del recorrido, y no habia con que cerrar cual de las
    dos cosas era porque los pivotes no se registraban.  El limite declarado
    es +-0.6 rad (ver rocker_pivot_zero_note en sim_config.yaml).
    """
    return [('suspension_{}'.format(j), 'rad',
             'Passive suspension joint {} angle'.format(j)) for j in joints]


def effort_columns(joints):
    """Par aplicado en cada articulacion motriz, del mismo JointState.

    Existen para contrastar de donde sale el enrollamiento del balancin.  Las
    ruedas van con un PID de velocidad de i_clamp 20.0 sobre un limite de par
    de 20.0 N*m, o sea que el integrador puede aparcarse en el maximo; y las
    dos ruedas de un bogie reaccionan sobre su brazo en el mismo sentido, hasta
    40 N*m contra los 15-24 N*m de momento que dan las cargas.  Medido el
    2026-09-11: al subir el tope del bogie a 1.0 rad, rev_4 se enrollo hasta el
    tope nuevo en el metro 17.7 y se quedo clavado 25 m con una rueda en el
    aire, EN LLANO.  Con esta columna se ve si el par estaba en su tope
    mientras pasaba.
    """
    return [('effort_{}'.format(j), 'N.m',
             'Applied torque at driven joint {}'.format(j)) for j in joints]


def velocity_columns(joints):
    """Velocidad de giro de cada rueda motriz, del mismo JointState.

    Existen porque con solo el par no se puede cerrar de donde sale el reparto
    desigual de traccion.  Medido el 2026-09-11 sobre
    metrics_rocker_bogie_par_ruedas: la rueda trasera de cada bogie va +3 N*m
    por encima de su media, CONSTANTE en llano, subiendo, bajando y girando, y
    plano en todos los deciles de carga de la media.  Con el par solo hay tres
    explicaciones indistinguibles -- se le manda mas rapido, no sigue la
    consigna, o patina -- y las tres piden lo mismo: la velocidad de junta
    contra la consigna.  Sale del mismo mensaje que la posicion y el par, asi
    que no hay suscriptor ni topic nuevo.
    """
    return [('velocity_{}'.format(j), 'rad/s',
             'Measured joint velocity at driven joint {}'.format(j))
            for j in joints]


def csv_header(columns):
    """['timestamp[s]', 'pitch[rad]', ...]"""
    return ['{}[{}]'.format(name, unit) for name, unit, _ in columns]


def load_gt_highrate(metrics_path):
    """gt_highrate.csv del mismo directorio que este metrics.csv, o None.

    POR QUE EXISTE.  La agitacion y la vibracion se median en ventanas de 1 s
    sobre metrics.csv, que va a 50 Hz.  Ninguna de las dos cosas era una
    eleccion: la ventana no podia bajar porque a 50 Hz una de 100 ms tiene 5
    muestras -justo el corte de m.sum() < 5- y la fuente era la que habia.

    Pero 1 s no corresponde a nada del sistema.  El LiDAR barre a 10 Hz, o sea
    100 ms por nube, y la variante troceada emite a 400 Hz; una ventana de 1 s
    promedia DIEZ barridos y borra justo la variacion por barrido que produce
    la distorsion de movimiento.

    gt_highrate.csv resuelve las dos a la vez: es verdad-terreno pura -sin
    ruido de sensor, sin sesgo, sin el termino de gravedad- y desde el
    2026-08-31 va a 1000 Hz de verdad (antes perdia el 43 % de los mensajes).
    A esa cadencia una ventana de 100 ms lleva 100 muestras.  Trae ademas
    gt_ax/gt_ay/gt_az, la aceleracion propia en ejes del cuerpo.

    Devuelve None si la corrida es anterior a que el logger lo escribiese; el
    llamante cae entonces a metrics.csv y a la ventana de 1 s.
    """
    import pandas as pd

    d = os.path.dirname(os.path.abspath(metrics_path))
    p = os.path.join(d, 'gt_highrate.csv')
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    renamed = {}
    for col in df.columns:
        base, _unit = strip_unit(col)
        renamed[col] = base
    df = df.rename(columns=renamed)
    if 'gt_ax' not in df.columns:
        return None            # corrida vieja: pose si, aceleracion no
    df.attrs['source'] = p
    return df


_VIB_AVISADO = set()


def vibration_magnitude(df, etiqueta=''):
    """|gt_a|, muestra a muestra.  NaN si la corrida no trae gt_a*.

    UNA SOLA DEFINICION DE VIBRACION, Y ES LA DE VERDAD-TERRENO.
    ===========================================================
    gt_a* es la aceleracion propia derivada del twist de Gazebo y rotada a
    ejes del cuerpo: sin ruido de sensor, sin sesgo y sin termino de gravedad.

    EL IMU NO VALE COMO SUSTITUTO, y por dos razones independientes.  Lo que
    habia antes en cuatro sitios era `|(a_x, a_y, a_z - 9.81)|`:

    1. Restar 9.81 de z solo quita la gravedad con el chasis nivelado.  En la
       ruta de la mina el cabeceo tiene sigma ~9 deg, asi que g se proyecta en
       los tres ejes y el residuo que queda dentro tiene un RMS de 1.84-1.95
       m/s^2 - MAS que toda la vibracion real del husky, que es 1.15.  La
       cifra medida asi es sobre todo gravedad.
    2. Aunque se quite la gravedad bien, con la actitud, el IMU sigue sin ser
       comparable entre plataformas: va montado en un punto distinto de cada
       robot, asi que mide a_cm + alpha x r + omega x (omega x r) con un brazo
       distinto en cada uno, mas su ruido sembrado.

    Medido sobre metrics_output_rtf025_3robots, las dos cosas juntas INVIERTEN
    el orden husky/rocker:

        gt_a          husky 1.153 < rocker 2.110 < tracked 2.423
        IMU |a-9.81z| rocker 2.045 < husky 2.412 < tracked 3.698
        IMU sin g     rocker 0.871 < husky 1.430 < tracked 3.110

    Por eso esto devuelve NaN -que las tablas imprimen como 'n/a'- en vez de
    caer al IMU: una casilla vacia es honesta, un numero del IMU no lo es.
    """
    import numpy as np

    faltan = [c for c in ('gt_ax', 'gt_ay', 'gt_az') if c not in df.columns]
    if faltan:
        clave = etiqueta or str(df.attrs.get('source', '?'))
        if clave not in _VIB_AVISADO:
            _VIB_AVISADO.add(clave)
            sys.stderr.write(
                'AVISO: %s no trae %s; la vibracion sale n/a. Es una corrida '
                'anterior a que el logger escribiese la aceleracion de '
                'verdad-terreno, y el IMU no sirve de sustituto (ver '
                'metrics_io.vibration_magnitude).\n' % (clave, ', '.join(faltan)))
        return np.full(len(df), np.nan)
    return np.sqrt(df['gt_ax'].values ** 2 + df['gt_ay'].values ** 2
                   + df['gt_az'].values ** 2)


def vibration_rms(df, etiqueta=''):
    """RMS de |gt_a| sobre todas las filas que se le pasen.

    El llamante decide el alcance: la corrida entera (recortado el warmup) da
    la cifra de las tablas, un trozo da la de esa ventana.  Que las dos salgan
    de la misma funcion es justo el punto: antes habia cuatro copias y dos de
    ellas leian el IMU.

    OJO AL COMPARAR ALCANCES.  El RMS global NO es la media de los RMS por
    ventana: por Jensen la media de ventanas es siempre menor, y el factor
    depende de lo a rafagas que sea la senal (x0.73 husky, x0.57 rocker, x0.81
    tracked con ventanas de 100 ms).  Son dos estadisticos distintos de la
    misma senal, no uno reescalado del otro.
    """
    import numpy as np

    a = vibration_magnitude(df, etiqueta)
    if not np.isfinite(a).any():
        return float('nan')
    return float(np.sqrt(np.mean(a[np.isfinite(a)] ** 2)))


def ventanas_alta_frecuencia(hr, t0, window_s):
    """Agitacion y vibracion por ventana, desde gt_highrate.

    Devuelve {inicio_de_ventana: (agitacion_rad, vibracion_m_s2)} con la
    agitacion como hypot(std(cabeceo), std(balanceo)) dentro de la ventana y
    la vibracion como RMS del modulo de gt_a*, la misma definicion que
    vibration_rms() aplicada a la ventana.

    Las ventanas se indexan por su instante de inicio RELATIVO a t0, que es
    el que usa el llamante sobre metrics.csv, para que las dos series se
    puedan casar aunque vengan de ficheros distintos.

    gt_highrate tiene huecos -mensajes que ODE no llego a publicar-, asi que
    una ventana con menos de 20 muestras se descarta en vez de promediarse
    con lo poco que haya.
    """
    import numpy as np

    t = hr['timestamp'].values - t0
    p = hr['gt_pitch'].values
    r = hr['gt_roll'].values
    a = vibration_magnitude(hr, 'gt_highrate.csv')
    if len(t) == 0:
        return {}
    bordes = np.arange(0.0, t.max() + window_s, window_s)
    idx = np.digitize(t, bordes) - 1
    out = {}
    for k in range(len(bordes) - 1):
        m = idx == k
        n = int(m.sum())
        if n < 20:
            continue
        out[round(float(bordes[k]), 6)] = (
            float(np.hypot(np.std(p[m]), np.std(r[m]))),
            float(np.sqrt(np.mean(a[m] ** 2))),
            n)
    return out


def strip_unit(column_name):
    """'pitch[rad]' -> ('pitch', 'rad');  'pitch' -> ('pitch', '')"""
    if column_name.endswith(']') and '[' in column_name:
        base, unit = column_name.rsplit('[', 1)
        return base.strip(), unit[:-1]
    return column_name.strip(), ''


# Columns whose name changed when units were introduced.  Older CSVs are
# remapped on load so previously collected runs stay usable.
LEGACY_ALIASES = {
    'pitch_rad': 'pitch',
    'roll_rad': 'roll',
    'yaw_rad': 'yaw',
    'slam_processing_time_s': 'slam_processing_time',
    'slam_icp_rms': 'slam_icp_inliers_ratio',
}


def load_metrics(path):
    """Read a metrics CSV into a DataFrame with unit-free column names.

    Units end up in df.attrs['units'] as {column: unit}.
    """
    import pandas as pd

    df = pd.read_csv(path)
    units = {}
    renamed = {}
    for col in df.columns:
        base, unit = strip_unit(col)
        base = LEGACY_ALIASES.get(base, base)
        renamed[col] = base
        units[base] = unit
    df = df.rename(columns=renamed)
    df.attrs['units'] = units
    df.attrs['source'] = os.path.abspath(path)
    return df


def unit_of(df, column, default=''):
    return df.attrs.get('units', {}).get(column, default)


def write_units_reference(columns, path):
    """Emit a machine-readable data dictionary next to the CSVs."""
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['column', 'unit', 'description'])
        for name, unit, desc in columns:
            w.writerow([name, unit, desc])


# ---------------------------------------------------------------------------
# TUM trajectory format:  timestamp tx ty tz qx qy qz qw
# The de-facto interchange format for trajectory evaluation, so the logged
# trajectories can be fed straight to evo / TUM tools for cross-checking the
# ATE and RPE numbers produced here.
# ---------------------------------------------------------------------------
TUM_HEADER = ('# timestamp[s] tx[m] ty[m] tz[m] qx[-] qy[-] qz[-] qw[-]\n')


def write_tum(path, rows):
    with open(path, 'w') as f:
        f.write(TUM_HEADER)
        for r in rows:
            f.write('%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n' % tuple(r))


def read_tum(path):
    import numpy as np

    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            data.append([float(v) for v in line.split()])
    return np.array(data)
