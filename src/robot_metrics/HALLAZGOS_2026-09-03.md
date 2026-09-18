# Por qué el differential parecía hacer mejor SLAM

Anotado el 2026-09-03. Punto de partida: *"todos tienen la misma trayectoria,
los mismos sensores y los mismos parámetros de RTAB-Map, ¿por qué el
differential sale mejor?"*

Respuesta corta: **casi todo lo que separa a las tres plataformas en la tabla
publicada lo meten el sensor troceado y el ordenador, no la locomoción.** Lo
que queda después de descontar los dos es un factor ~2, no el ~6 del titular.

Este documento resuelve el punto 1 de `PENDIENTES.md` (el confundido de la
velocidad) y abre uno nuevo (el confundido de la tasa de proceso).

---

## Resumen

| # | Hallazgo | Estado |
|---|---|---|
| 1 | La columna del titular mide `/rtabmap/odom`, no el grafo. En el grafo optimizado las siete corridas caben en 6 cm. | medido |
| 2 | El sesgo de rumbo lo mete el LiDAR troceado. Con el monolítico las tres plataformas son indistinguibles. | medido, A/B de sensor |
| 3 | **NO es la velocidad.** Igualarla no transfiere la inmunidad del husky. | medido, A/B cruzado |
| 4 | Las tres corren el front-end a tasas distintas (5.1 / 6.4 / 9.6 Hz) que fija la MÁQUINA. Controlarlo baja la diferencia de ~5.5x a ~2x. | medido |
| 5 | El husky no es inmune, es menos susceptible. Su error por nube es ~3x menor y de signo contrario. | medido |
| 6 | Por qué el signo es distinto en el husky. | **abierto** |

Todo lo de aquí es **n = 1 por celda**. Nada entra en el paper sin repeticiones.

---

## 1. La campaña mide odometría, no SLAM

`metrics.csv` registra `/rtabmap/odom` (`metrics_logger.py:174`,
`metrics_logger.launch:9`): la salida de `icp_odometry`, sin ningún cierre
aplicado. De ahí sale `ate_origin_trans_rmse`, la columna del titular.

Con `evaluate_run` de la propia `aggregate_runs.py`, corridas enteras:

| corrida | `ate_origin` | `ate_umeyama` | **`ate_opt`** |
|---|---|---|---|
| Husky troceado (2 vueltas) | 1.842 | 0.504 | **0.112** |
| Rocker troceado (2 vueltas) | 11.320 | 5.620 | **0.174** |
| Husky troceado B | 2.763 | 0.265 | **0.122** |
| Tracked troceado | 9.255 | 3.339 | **0.160** |
| Husky monolítico | 2.267 | 0.536 | **0.111** |
| Rocker monolítico | 2.886 | 0.735 | **0.115** |
| Tracked monolítico | 3.313 | 0.489 | **0.129** |

**En `ate_opt` las siete caben en 6 cm.** El optimizador borra la deriva de
rumbo entera: rocker troceado **+50.7° → −0.85°**, tracked **+32.1° → −0.68°**.
Deriva final de posición del grafo: 6–18 cm en 132 m, las siete.

**Qué hacer.** Decidir qué afirma el paper. Si es SLAM, la columna es
`ate_opt_trans_rmse` y dice que las plataformas empatan. Si es localización en
línea, `ate_origin` vale pero hay que llamarla por su nombre — odometría ICP —
y sigue contaminada por lo de abajo.

**Trampa.** El ATE alineado al origen sobre `opt_traj.tum` castiga un
desalineamiento inicial de un grafo que luego es casi perfecto (el tracked sale
4.28 m con 0.18 m de error final). Para el grafo hay que usar la alineación
rígida sobre todo el tramo, que es lo que `optimised_ate` ya hace.

---

## 2. El sesgo lo mete el sensor, no la locomoción

A/B de sensor, misma ruta, misma configuración, cortado a 132 m:

| corrida | ATE | deriva final | rumbo | deg/m | RPE |
|---|---|---|---|---|---|
| Husky troceado | 1.53 | 0.86 | −3.3 | −0.025 | 0.094 |
| Rocker troceado | 6.27 | 9.63 | **+22.1** | **+0.167** | 0.107 |
| Tracked troceado | 9.23 | 12.46 | **+31.2** | **+0.236** | 0.100 |
| Husky monolítico | 2.27 | 2.35 | −0.9 | −0.007 | 0.062 |
| Rocker monolítico | 2.88 | 4.10 | −1.6 | −0.012 | 0.058 |
| Tracked monolítico | 3.32 | 2.52 | −0.8 | −0.006 | 0.070 |

Con LiDAR monolítico las tres son indistinguibles en rumbo: −0.007, −0.012,
−0.006 deg/m. Y el troceado cuesta lo mismo a las tres en precisión local
(RPE x1.43 a x1.85). O sea: **el troceado las castiga por igual localmente, y
sólo dos de ellas convierten ese error en deriva de rumbo.**

### La configuración troceada está verificada idéntica

`check_sim_parity.py` **no audita la variante troceada** (no menciona `sliced`,
`wedge` ni `rotor` ni una vez): audita los modelos monolíticos. Comprobado a
mano generando los tres modelos:

- Cuña idéntica en las tres: `update_rate` 400.0000, 110 muestras, ±0.183260 rad
  (21.000°), 16 anillos de ±0.261799 rad, rango 0.3–101.0 m, ruido gaussiano
  sd 0.011314, `min_range` 0.9 del plugin.
- Rotor idéntico: eje `0 0 1`, damping 0.0, friction 0.0, masa 0.01, izz 1e-05.
  `revolute` con límites ±1e16 en la rama SDF y `continuous` en la URDF:
  equivalentes.
- Velocidad real del rotor: 62.8309–62.8327 rad/s, sd 0.006–0.010, deriva
  ≤15 ppm, cero reimpulsos, mismo signo.
- Nubes equivalentes: monolítico 28108 ± 5 puntos y 0° de huecos; troceado
  27412–27779 y 1.1–5.9° de huecos, con el mismo perfil de alcances
  (p05 1.78–1.83, mediana 4.39–4.46, p95 17.04–17.15) y los mismos 15 anillos.
- **Autoeco descartado por sector de azimut**, no por el mínimo global: el
  perfil de `r_min` y de puntos <2 m coincide sector a sector entre troceado y
  monolítico en las tres (deltas ±0.1 m).

**Pendiente:** añadir la variante troceada a `check_sim_parity.py`.

---

## 3. NO es la velocidad — resuelve el punto 1 de PENDIENTES.md

El punto 1 de `PENDIENTES.md` señalaba, con razón, que el husky va más lento y
que eso le regala barridos por metro. Está resuelto: **la velocidad es real,
tiene una causa identificada, y no explica la deriva.**

### Por qué el husky va más lento

No es que no alcance lo que se le manda: **lo supera en un 2 %.** Lo que es más
bajo es el mandato.

| corrida | \|xte\| medio | cmd medio | v real | real/cmd | nubes/m |
|---|---|---|---|---|---|
| Husky (troc A / B / mono) | 0.102 / 0.105 / 0.101 | 0.409 / 0.408 / 0.411 | 0.417 / 0.416 / 0.418 | **1.02** | 24.0 |
| Rocker (troc / mono) | 0.026 / 0.024 | 0.459 / 0.462 | 0.456 / 0.459 | 0.99 | 21.9 |
| Tracked (troc / mono) | 0.034 / 0.036 | 0.456 / 0.454 | 0.459 / 0.457 | 1.01 | 21.8 |

`trajectory_follower.py` manda `v = v_max/(1 + curve_slowdown*|curv|)`, y a
`|curv|` le suma `k_xte`(2.5) por el error lateral, con tope `xte_curv_max` 0.5.
El husky sigue la línea peor (0.10 m contra 0.03) → el seguidor lo lee como
curvatura → le manda 0.41 en vez de 0.46. Con xte 0.102 sale v 0.434, con 0.026
sale 0.481: razón predicha 0.902 contra la medida 0.891, **cuadra al 1 %**.

**La velocidad no es un parámetro que la campaña fije: es una SALIDA del
seguimiento de cada plataforma.** "Misma trayectoria" son los mismos waypoints,
no el mismo perfil de velocidad ni la misma línea.

### El A/B cruzado

Con `--max-vel-x` (opción nueva), LiDAR troceado, misma ruta, cortado a 132 m:

| corrida | v | nubes/m | giro/m | avance/barrido | **deriva** |
|---|---|---|---|---|---|
| Husky troceado | 0.417 | 23.97 | 7.73 | 41.7 mm | **−0.021** |
| Husky troceado B | 0.416 | 24.04 | 7.96 | 41.6 mm | **−0.010** |
| **Husky RÁPIDO** (v_max 0.560) | **0.471** | **21.25** | 7.02 | **47.1 mm** | **−0.019** |
| Rocker troceado | 0.456 | 21.93 | 9.39 | 45.6 mm | **+0.191** |
| **Rocker LENTO** (v_max 0.445) | **0.408** | **24.53** | 9.10 | **40.8 mm** | **+0.259** |
| Tracked troceado | 0.459 | 21.80 | 4.24 | 45.9 mm | **+0.230** |

**Las dos plataformas se intercambiaron el régimen cinemático y cada una se
quedó con SU deriva.** El husky a 47.1 mm/barrido —más que el rocker original—
sigue en −0.019. El rocker a 40.8 mm/barrido —menos que el husky original— no
sólo no mejora: pasa de +0.191 a +0.259.

Tampoco se salva midiendo por barrido: el rocker pasa de 0.00869 a 0.01057
deg/nube. Ir más despacio lo empeora en las dos monedas.

**La manipulación fue limpia:** se tocó sólo `max_vel_x`, nada del lazo. El giro
por metro del rocker se mantuvo (9.39 → 9.10) y su error lateral también
(0.026 → 0.024). En el husky el xte incluso bajó (0.102 → 0.089) porque el
lookahead crece con la velocidad.

**No se usó `--k-xte 0`** aunque también habría acelerado al husky: `k_xte`
alimenta `curv`, que va a la vez al freno **y al volante**
(`w = curv*max(v, w_v_floor)`), así que anularlo le quita la corrección lateral
de la dirección. El propio código documenta que con la ganancia vieja el husky
tardaba 25.9 m en recuperar 0.90 m de desvío y llegó a volcar.

### Lo demás que quedó descartado, midiendo

- **Brazo de palanca del LiDAR**: el tracked lo tiene en 0.0000 m, exactamente
  como el husky, y es el de peor sesgo. El rocker, con 0.3063 m, queda en medio.
- **Actitud media del cuerpo**: \|tilt\| mediano 10.40 / 9.97 / 10.49° —
  idénticos. No hay cono de barrido asimétrico.
- **Velocidad de giro**: el tracked gira la mitad (1.94 contra 3.16 y 4.24
  deg/s) y deriva el doble.
- **Vibración, velocidades angulares laterales, montaje del sensor.**
- **Cierres de bucle**: en las nueve bases de datos hay **cero** cierres
  globales; sólo `LocalSpaceClosure` y enlaces de gravedad, que fijan alabeo y
  cabeceo pero no rumbo. Su densidad por nodo tampoco ordena.
- **La ruta**: separación media punto a punto al mismo avance, entre
  cualesquiera dos plataformas, 0.01–0.56 m.

---

## 4. Confundido nuevo: la máquina fija la tasa del front-end

El ensamblador entrega **10 nubes por segundo de simulación** a las tres. Lo que
cada una llega a **procesar** no es lo mismo:

| corrida | RTF | odom [Hz] | procesa | nubes/m | deriva [deg/m] | **error/nube** |
|---|---|---|---|---|---|---|
| Husky troceado | 0.482 | 5.09 | 51 % | 12.08 | −0.0195 | **−0.00162** |
| Husky troceado B | 0.484 | 5.36 | 54 % | 12.74 | −0.0097 | **−0.00076** |
| Husky RÁPIDO | 0.489 | 5.75 | 58 % | 12.07 | −0.0223 | **−0.00185** |
| Husky RTF 0.25 | 0.250 | 8.26 | 83 % | 19.84 | −0.0946 | **−0.00477** |
| Husky RTF 0.10 | 0.100 | 6.91 | 69 % | 16.52 | −0.1599 | **−0.00968** |
| Rocker troceado | 0.377 | 6.41 | 64 % | 14.10 | +0.1937 | **+0.01374** |
| Rocker LENTO | 0.401 | 7.61 | 76 % | 18.72 | +0.2685 | **+0.01434** |
| Tracked troceado | 0.246 | 9.56 | 96 % | 20.85 | +0.2286 | **+0.01096** |

*(todo cortado a 120 m, porque la corrida a RTF 0.25 se pasó del tope de 1200 s
y archivó 123.69 m)*

**EL SIGNO VA AL REVÉS DE LO QUE PARECE.** Con `use_sim_time`, un simulador
LENTO le regala al SLAM más tiempo de **pared** por segundo de simulación. La
plataforma de física más cara (tracked, RTF 0.246) procesa el 96 % de sus nubes;
la más barata (husky, RTF 0.482) tira la mitad. El modelo cuadra:
capacidad = (1/t_ICP)/RTF, con t_ICP mediano 0.16–0.28 s, topada en 10 nubes/s.

**Con el monolítico el error por nube es ~0 en las tres** (−0.0003 a −0.0006)
aunque el tracked procese 21.9 nubes/m. La tasa sólo **amplifica** el error que
mete el sensor troceado; no lo crea.

### El error por nube es invariante frente a la velocidad

El rocker da +0.01358 y +0.01388 deg/nube con derivas por metro que se llevan un
36 %. Eso convierte el A/B de velocidad en una predicción cuantitativa:

    deriva = error_por_nube x nubes_por_metro
    Rocker LENTO:  predicho +0.2536,  medido +0.2592  ->  2.2 % de error

Frenar al rocker le subió la deriva **porque le subió las nubes/m**
(14.04 a 18.68), no por la velocidad.

### Por qué importa para el paper

La tasa del front-end la fija el ORDENADOR, no la locomoción, y no está
controlada. El tracked procesa 1.7x más nubes por metro que el husky. Comparando
a tasa igualada (~19–21 nubes/m):

| | nubes/m | ATE |
|---|---|---|
| Husky RTF 0.25 | 19.83 | **4.20** |
| Rocker LENTO | 18.82 | 8.26 |
| Tracked troceado | 20.83 | 8.72 |

**Controlando la tasa, la diferencia entre plataformas pasa de ~5.5x a ~2x.**
Más de la mitad del titular era el ordenador.

---

## 5. El husky no es inmune: es menos susceptible

Bajándole el techo de RTF, el husky sí deriva y sí sube su ATE:

| corrida | ATE | **ATE con rumbo GT** | queda | escala avance | temblor lat. | RPE | rumbo |
|---|---|---|---|---|---|---|---|
| Husky troceado | 1.59 | 0.26 | 17 % | 1.0029 | 10.1 mm | 0.0870 | −2.46° |
| Husky troceado B | 2.91 | 0.41 | 14 % | 1.0031 | 10.3 mm | 0.0847 | −2.69° |
| Husky RÁPIDO | 2.43 | 0.24 | 10 % | 1.0017 | 11.0 mm | 0.0847 | −3.62° |
| Husky RTF 0.25 | 4.20 | 0.30 | 7 % | 0.9965 | 10.1 mm | 0.0749 | −12.75° |
| Husky RTF 0.10 | 6.44 | 0.42 | 6 % | 1.0043 | 14.2 mm | 0.0898 | −19.32° |
| Rocker troceado | 5.75 | 0.54 | 9 % | 0.9958 | 20.1 mm | 0.1058 | +25.33° |
| Rocker LENTO | 8.26 | 0.85 | 10 % | 0.9901 | 20.5 mm | 0.1079 | +32.58° |
| Tracked troceado | 8.72 | 0.52 | 6 % | 1.0085 | 17.6 mm | 0.0991 | +28.28° |

**El ATE es rumbo, íntegro, en las ocho.** Reintegrando los pasos que el propio
front-end midió pero con el rumbo verdadero, sobrevive el 6–17 %. No hay ningún
modo de error de traslación: la escala de avance es 0.99–1.01 en todas.

Lo que separa al husky no es inmunidad: es un **error por nube ~3x menor y de
signo contrario**, más el hecho de que en la campaña tiraba la mitad de las
nubes. Es lo único que sigue sin explicación.

### `--rtf` no es un mando limpio por debajo de 0.25

A RTF 0.10 el ensamblador recibe **62.5 rebanadas por vuelta** en vez de las
39–41 de todas las demás corridas: con tiempo de pared de sobra, Gazebo
**supera** el `update_rate` de 400 Hz que se le pide al `gpu_ray`. El
ensamblador, que es Python, se convierte en el cuello de botella y publica menos
nubes — el front-end recibe 6.91/s con su t_ICP más bajo de siempre (0.162 s),
o sea **hambriento, no ahogado**.

La corrida a RTF 0.10 cambió por tanto dos cosas a la vez y **no extiende
limpiamente la curva**. Hay un punto de inflexión hacia RTF 0.25:

- **de 0.50 a 0.25 gana el SLAM** (51 % a 83 % de nubes; sensor sin cambios,
  39 a 41 rebanadas): manipulación razonablemente limpia.
- **por debajo de 0.25 gana el sensor** y ahoga al ensamblador: la tasa de
  proceso BAJA en vez de subir.

**Buena noticia:** a los RTF nativos las tres plataformas reciben 39, 40 y 41
rebanadas por vuelta. El desbordamiento sólo aparece en corridas forzadas por
debajo de 0.25, así que no contamina la campaña.

---

## Herramientas nuevas

En `run_campaign.sh`, todas vacías por defecto — sin ellas la campaña no cambia
en nada, y cualquiera de ellas marca la corrida `paper_usable: false` y deja
constancia en `follower_overrides` del `run_meta.yaml`:

| opción | qué hace |
|---|---|
| `--max-vel-x V` | techo de velocidad del seguidor |
| `--k-xte K` | ganancia de corrección lateral (**ojo**: va también al volante) |
| `--rtf R` | techo de factor de tiempo real, en caliente |

`scripts/set_rtf.py` fija el techo por `/gazebo/set_physics_properties`. **No
edita `lcmine.world` a propósito**: `check_sim_parity.py` comprueba que los tres
mundos son byte a byte idénticos por md5. Reenvía el `ode_config` completo — si
se omitiera, el solver volvería a los 50 iteraciones compilados por defecto en
vez de los 100 que declara el mundo, y eso sí cambiaría la física. Verifica
después de escribir y aborta si el techo no quedó donde se pidió.

Scripts de análisis, en `~`: `comparar_todo.py`, `diag_retardo.py`,
`diag_carga.py`, `diag_carga2.py`, `diag_ensamble2.py`, `ab_resultado.py`,
`ab_control.py`, `rtf_resultado.py`, `rtf_ate.py`, `cmp_lidar_cfg.py`,
`cmp_rotor.py`, `cmp_nubes.py`, `cmp_autoocl.py`, `cmp_ruta.py`, `diag_xte.py`.

## Trampas encontradas

- **El tope de 1200 s no basta por debajo de RTF 0.4.** La corrida a RTF 0.25 se
  cortó y archivó 123.69 m. Usar `--duration 4200` con `--rtf 0.10`.
- **`plot_position_matched.py` peta con un solo robot** por campaña
  (`numpy.AxisError: axis 1 is out of bounds`): compara plataformas y necesita
  dos o más. Es posterior a `metrics.csv`, los `.tum` y la base; la campaña sale
  con estado 0. No afecta a ningún número.
- **`rosrun` no encuentra los scripts de `scripts/`**: sólo mira en `devel/lib`
  y `devel/share`. Invocar por ruta absoluta.
- **Mezclar `>` y `>>` sobre el mismo log** desde dos procesos: cada uno lleva su
  desplazamiento y el que abrió con `>` sobrescribe al otro. `set_rtf.py`
  escribía en `gazebo.log` y perdía su veredicto; ahora va a `set_rtf.log`.
- **`Data.scan` de `rtabmap.db`** es zlib crudo desde el byte 0, 4 float32 por
  punto. Sin cabecera de `cv::Mat`.
- **El `cleanup()` de `run_campaign.sh` hace `pkill -f`** sobre los nombres de
  los procesos de Gazebo y ROS, y `-f` casa contra la línea de órdenes entera:
  cualquier shell de vigilancia que los mencione se suicida. Poner el filtro en
  un archivo de script, no en la línea de órdenes.

## Qué queda

1. **Repetir todo con n ≥ 3.** Todas las celdas de aquí son una sola corrida.
2. **Igualar la tasa de entrada** con un `topic_tools/throttle` entre el
   ensamblador e `icp_odometry`, a la tasa de la más lenta. Es la vía limpia: no
   toca ni la física ni el sensor, al contrario que `--rtf`.
3. **Añadir la variante troceada a `check_sim_parity.py`.**
4. **La pregunta abierta:** por qué el error por nube del husky es negativo y
   pequeño (−0.0016) y el del rocker y el tracked positivo y grande (+0.011 a
   +0.014). El experimento que discrimina la única familia que queda —
   acoplamiento de quiralidad entre el giro del rotor y el del vehículo — es
   invertir el sentido del rotor con la ruta igual. Si el sesgo no cambia de
   signo, esa familia queda descartada entera.
