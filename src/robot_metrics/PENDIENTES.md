# Pendientes abiertos tras la campaña de LiDAR troceado

Anotado el 2026-08-27, sobre la campaña `fixed_trajectory / sliced_lidar`
(1 corrida por plataforma, 2 vueltas, semilla 1). Los números de aquí están
medidos en esa campaña salvo donde se diga otra cosa.

Contexto de por qué existen estos tres puntos: el análisis emparejado por
posición (`plot_position_matched.py`) dio **nulo** — r = −0.154, IC95 %
[−0.350, 0.050], y la plataforma más agitada del bin es también la de más error
en 17 de 69 bins (24.6 %, por debajo del 33.3 % del azar). Los tres puntos de
abajo son las razones por las que ese nulo **todavía no significa** que la
vibración no afecte al SLAM.

---

> **ACTUALIZACION 2026-09-03.** El punto 1 esta RESUELTO y el punto 2
> reformulado: ver `HALLAZGOS_2026-09-03.md`. En corto: la velocidad del
> Husky es real y tiene causa identificada (su propio error lateral
> realimenta el termino de curvatura del seguidor), pero **no explica la
> deriva de rumbo** — igualarla en un A/B cruzado no transfiere la ventaja.
> Lo que si ordena el resultado es un confundido que no estaba en esta
> lista: las tres plataformas corren el front-end a tasas distintas
> (5.1 / 6.4 / 9.6 Hz) que fija la MAQUINA, no la locomocion.

> **ACTUALIZACION 2026-09-04.** Ese confundido tiene DOS fuentes, no una, y
> hasta hoy solo se conocia la primera.
>
> 1. **rtabmap saturado.** Manda a RTF ~0.4-0.5: `icp_odometry` usa 1.0-1.2 de
>    su presupuesto de pared por nube y tira entre el 35 y el 49 % de ellas,
>    con el histograma de huecos dominado por k=2 -una si, una no-.
> 2. **El ensamblador descartaba vueltas en silencio.** Aparece cuando la
>    primera desaparece. A RTF 0.25 el `icp_odometry` baja a 0.45 de uso y aun
>    asi se perdia el 18.5 / 13.3 / 7.2 % (husky / rocker / tracked). No era
>    rtabmap: el `delay` de la nube que sigue a un hueco es igual al de una
>    normal (151 vs 153 ms) y el `update time` de la anterior tampoco sube. Las
>    nubes **no llegaban**. `rotor_sweep_assembler.vuelta()` consumia las cunas y
>    `procesa()` comprobaba DESPUES si el encoder cubria la ventana; con el
>    encoder a 200 Hz contra cunas a 400 Hz eso falla ~25 % de las veces, y el
>    reparto depende de como entrelaza gzserver sus hilos, o sea del RTF y del
>    robot.
>
> **ARREGLADO el 2026-09-04**: la comprobacion se hace en `vuelta()`, antes de
> consumir las cunas, y si el encoder no ha llegado se ESPERA 2 ms en vez de
> tirar la vuelta. El nodo lleva contadores
> (`formadas / publicadas / desc_* / esperas`) que salen en cada linea
> periodica de `gazebo.log` y que `run_campaign.sh` copia al `run_meta.yaml`
> como bloque `sliced_lidar:`, con un aviso si se pierde mas del 1 %. Banco
> offline en `test/test_rotor_sweep_assembler.py`, que corre el mismo flujo de
> mensajes contra las dos versiones de `vuelta()`.
>
> **Que esperar de la proxima campaña.** A RTF 0.25 el uso de `icp_odometry`
> por nube no cambia (0.45 antes y despues: los huecos eran tiempo ocioso, no
> alivio), asi que las tres deberian entregar ~9.88 Hz -no 10: la ventana
> avanza 100 ms mas medio periodo de cuna-. **Ojo con la lectura:** el
> differential pasa de 8.04 a ~9.88 Hz, o sea de 19.6 a ~24 nubes/m, y si el
> mecanismo deriva/m = error/nube x nubes/m se sostiene, su ATE deberia
> EMPEORAR, no mejorar. El arreglo hace la medida comparable, no mejor.
>
> A RTF ~0.48 este arreglo no sirve de nada: alli manda la fuente 1
> (uso 1.14, y 1.22 con la tasa ya corregida). La campaña va a RTF 0.25.

## 1. El control del Husky: va más lento y eso le regala precisión

**RESUELTO el 2026-09-03: no es la causa.** Sigue siendo un confundido
real y hay que reportarlo, pero el A/B cruzado lo descarta como
explicacion. Ver `HALLAZGOS_2026-09-03.md`, seccion 3.

**Qué se midió.** Con trayectoria fija las tres recorren el mismo camino, pero
no a la misma velocidad:

| | distancia | duración | velocidad media | barridos/m | deformación por barrido | ATE |
|---|---|---|---|---|---|---|
| Husky | 288.1 m | 774.1 s | **0.372 m/s** | 26.9 | 36.9 mm | **7.94 m** |
| Rocker-bogie | 275.5 m | 614.4 s | 0.448 m/s | 22.3 | 44.2 mm | 11.54 m |
| Tracked | 277.4 m | 619.7 s | 0.448 m/s | 22.3 | 44.3 mm | 22.91 m |

**Por qué importa.** El Husky va un **17 % más despacio**, y eso le da dos
ventajas a la vez, ninguna de las cuales es una propiedad de su locomoción:

- **21 % más restricciones ICP por metro** (26.9 barridos/m contra 22.3).
- **La nube menos deformada**: la traslación durante los 100 ms de una vuelta
  del rotor es el término que domina el presupuesto de distorsión, y va con la
  velocidad.

O sea que la plataforma con mejor ATE es también la que recibió más datos y la
nube más limpia. **Es un confundido, y ahora mismo ordena el resultado.** Es el
mismo tipo de fallo que el montaje de sensores: algo que la trayectoria fija
debería controlar y no controla — iguala el *camino*, no la *velocidad*.

**Qué hacer.** Mirar cómo fija la velocidad `trajectory_follower.py` y si es
igualable entre plataformas. Si el seguidor satura contra algún límite del
Husky (par, velocidad de rueda, o el propio perfil de velocidad del seguidor),
hay que saberlo: si el Husky *no puede* ir a 0.448 m/s, entonces la velocidad
no es igualable y hay que reportar la comparación a velocidades distintas y
descontar el efecto, no fingir que no está.

**Lo que esto NO explica.** El tracked y el rocker van exactamente a la misma
velocidad y aun así se separan 2× en ATE (22.91 contra 11.54). Ahí queda una
diferencia real que la velocidad no cubre, y es la comparación limpia que hay
hoy.

---

## 2. Degradación del LiDAR por vibración: hay que modelarla, y falta la fuente

**Estado actual: no está modelada en absoluto.** Verificable por inspección —
los tres modelos declaran

```xml
<noise><type>gaussian</type><mean>0</mean><stddev>0.011314</stddev></noise>
```

y `<gaussianNoise>0.0</gaussianNoise>` en el plugin. Es una gaussiana **fija
sobre el rango**, y nada más. Las direcciones de los haces son exactas: no hay
error de apuntamiento, ni jitter del encoder, ni retornos perdidos. **No existe
ninguna vía de código por la que la aceleración del cuerpo cambie la calidad de
una medida.** La vibración solo puede entrar moviendo el sensor entre
rebanadas, y eso solo en modo troceado.

### El paper propuesto NO sirve para esto

**Lu, Fowler, Starek et al., *"Scan Pattern Characterization of Velodyne VLP-16
Lidar Sensor for UAS Laser Scanning"*, Sensors 2020, 20(24), 7351.**
<https://www.mdpi.com/1424-8220/20/24/7351>

Comprobado leyendo el texto completo (vía PMC, MDPI devuelve 403):

- Modela **solo el patrón geométrico de barrido** de la configuración en
  abanico del VLP-16, en condiciones idealizadas.
- Sus ecuaciones toman altura de vuelo, velocidad de avance, tasa de rotación,
  frecuencia de pulso, guiñada y separación angular entre canales (Δω = 2°).
  Salida: densidad de puntos p(x), separación óptima entre líneas de vuelo y
  posición de los huecos de cobertura.
- **La vibración no se menciona en ninguna parte del artículo.** Tampoco
  modela error de rango, ruido de medida, ni error de apuntamiento. Los propios
  autores listan como limitación que no cubren la geometría interior del sensor
  ni el sesgo de rango dependiente del ángulo de incidencia.

Conclusión: **no aporta el coeficiente que hace falta.** Puede citarse como
respaldo de que la densidad de retorno del VLP-16 es no uniforme y produce
huecos de cobertura que no se corresponden de forma simple con la velocidad de
avance — que es contexto útil para el argumento de muestreo — pero su geometría
es de UAV mirando hacia abajo desde altura h, no de robot terrestre en galería,
así que las ecuaciones no se trasladan.

### Qué haría falta de verdad

El mecanismo intrínseco del sensor, **independiente del montaje**, es el error
de apuntamiento: bajo vibración el azimut que reporta el encoder del rotor no
coincide con la dirección real del haz. A distancia R, un error angular δθ se
convierte en error lateral R·δθ.

- **Dónde va:** `rotor_sweep_assembler.py`. Ya recibe cada rebanada con el ángulo
  real del encoder y la de-rota; ese es el punto donde perturbar el azimut.
- **Cómo parametrizar:** σ_θ = k · a_rms, con **k idéntico en las tres
  plataformas** (k es propiedad del sensor; lo que cambia entre robots es su
  propia vibración). Sembrado desde la semilla de la campaña.
- **De dónde sacar k:** hace falta una medición de precisión angular bajo
  vibración controlada (mesa vibratoria) o un presupuesto de error de boresight
  de la literatura de mobile mapping. **La hoja de datos del VLP-16 no sirve**:
  su especificación de vibración es de supervivencia (MIL-STD-810G), no de
  rendimiento bajo vibración.

### El umbral que hay que comprobar ANTES de correr nada

Traducido a ángulo a 10 m:

| | error lateral a 10 m | equivale a |
|---|---|---|
| Ruido propio del VLP-16 | 11.3 mm | **0.065°** |
| Discretización hoy (~23 rebanadas/vuelta) | ~19 mm | 0.109° |
| Discretización a N = 40 | 10.9 mm | 0.062° |

Si σ_θ sale bastante por debajo de **0.06°**, vuelve a quedar enterrado bajo el
ruido del propio sensor y se repite el mismo no-resultado con más trabajo. El
orden correcto es: fijar k con una fuente → calcular σ_θ para las vibraciones
medidas (2.3–4.8 m/s² RMS) → **solo entonces** decidir si merece una campaña.

Y hay que aceptar de antemano el desenlace posible: que k salga de una fuente
real y el efecto quede en 0.01°. Eso sería un resultado legítimo y publicable
tal cual — *la degradación por vibración es real pero despreciable frente al
ruido de rango del sensor a estas amplitudes*. **Lo que no vale es subir k
hasta que se vea.**

### Descartado a propósito: complianza del mástil

Hoy el montaje está **controlado**: los tres cuelgan de una junta fija con
mástil sin masa, o sea rígidos e idénticos en lo dinámico. Modelar resonancia
del montaje introduciría una propiedad que **difiere por plataforma**, y la
comparación dejaría de aislar la locomoción. Se deja fuera deliberadamente.

---

## 3. Tasa de muestreo: hay tres, y ninguna es la que hace falta

**El problema.** La teoría dice que la distorsión la causa el cambio de pose
*durante* el barrido, que ocurre en milisegundos. La cadena lo muestrea así:

| eslabón | tasa | resuelve hasta |
|---|---|---|
| Física de Gazebo (`max_step_size` 0.0005) | 2000 Hz | 1000 Hz |
| Cuña del LiDAR troceado (**entregada**, no pedida) | 229.6 Hz | **115 Hz** |
| IMU | 100 Hz | 50 Hz |
| `metrics.csv` | 50 Hz | **25 Hz** |

Tres consecuencias, en orden de gravedad:

**(a) La métrica de vibración con la que se correlaciona ve hasta 25 Hz.**
"Vibration RMS" y "Peak |a_z − g|" salen de `metrics.csv`. Se está
correlacionando una vibración con ancho de banda 25 Hz contra un fenómeno cuyo
ancho de banda modelado es 115 Hz, dentro de una física que llega a 1000 Hz.
Si hay energía por encima de 25 Hz no está ausente del número: está **plegada
dentro** por aliasing. El pico de 44.1 m/s² del tracked es una muestra cada
20 ms de algo casi seguro más rápido — su valor depende de *cuándo* se miró.
**Es posible que la variable independiente no esté midiendo la cantidad que
causa el efecto.**

**(b) La cuña entrega 229.6 Hz de los 400 pedidos** (`rotor_hz` 10 × `slices`
40). La vuelta se discretiza en ~23 rebanadas efectivas, no 40 — el punto de
convergencia N ≈ 20, no el N = 40 que justificó el estudio de convergencia.
El giro entre rebanadas queda en 15.7° contra una cuña de 21°, así que **no se
abren huecos** (el margen hace su trabajo), pero el suelo de discretización se
queda en ~19 mm a 10 m en vez de 10.9 mm.
**Arreglo:** bajar el rotor manteniendo fija la razón ω/Ω, **no** subir el
`update_rate` — por encima de 400 Hz el `gpu_ray` se desboca y entrega de más.

**(c) Por encima de 115 Hz la nube no puede representar la vibración**, y peor:
la pliega. Dentro de una rebanada el sensor está congelado 4.36 ms. Un VLP-16
real dispara cada ~55 μs. Si el chasis tiene energía a 200 Hz, muestrearla a
229.6 Hz la alias a ~30 Hz y la nube acaba con un bamboleo que no existe. No es
un modelo conservador, es un modelo con un artefacto.

### Cómo zanjarlo: usar el ground truth de Gazebo, no el IMU

`/gazebo/model_states` **ya se publica a 2000 Hz** — una vez por paso de
física. Es `metrics_logger.py` quien lo baja a 50 Hz al registrar. No hay que
tocar el IMU ni la física, solo registrar sin estrangular.

Por qué el GT es el instrumento correcto:

- **El IMU tiene un suelo de ruido que taparía justo lo que se busca.**
  σ = 0.009 rad/s muestreado a 1000 Hz da una densidad espectral plana de
  1.6·10⁻⁷ (rad/s)²/Hz; el contenido real por encima de 100 Hz puede estar por
  debajo. Además el giroscopio cuantiza a 0.25 mrad/s (`<precision>`) y lleva
  sesgo estático y dinámico. El GT es el vector de estado tal cual sale de ODE.
- **Es la cantidad exacta**, no una observación derivada: la pose que usa cada
  rebanada es la del cuerpo.
- **20× de ancho de banda**: Nyquist 1000 Hz contra 50 Hz.

Dos cuidados:

- **`ModelStates` no lleva `header`** — lo documenta el propio
  `metrics_logger.py`. Cada muestra se sella al recibirla, con error acotado
  por un paso de física (0.5 ms). A 50 Hz es irrelevante; a 2000 Hz es **una
  muestra entera** de jitter. Se maneja asumiendo Δt = 0.5 ms uniforme y
  **verificando** el conteo de mensajes contra el tiempo de simulación.
- **Usar `base_link`, no el link del Velodyne.** En el rocker
  `velodyne_base_link` **no existe** en tiempo de ejecución: la conversión URDF
  lo fusiona en `base_link` (comprobado en el banco). Con montaje rígido da
  igual: la velocidad angular es idéntica y la lineal difiere por el brazo
  ω×r, con r = 0.900 m verificado.

### La prueba que responde en milímetros, no en Hz

Con la pose verdadera a 2000 Hz se puede contestar en las unidades que importan:

1. Registrar la pose real del sensor a 2000 Hz durante una corrida.
2. Reensamblar la nube dos veces: con la pose **continua** (2000 Hz ≈ la
   verdad) y con la **cuantizada a 229.6 Hz** que usa la simulación.
3. La diferencia a 10 m es **el error de modelado que introduce el
   submuestreo**.
4. Compararla con los 11.3 mm de ruido del propio sensor.

Si sale muy por debajo de 11.3 mm, la preocupación queda descartada con un
número. Si sale por encima, se sabe cuánto falta y si el remedio es más
rebanadas o un modelo distinto.

El espectro sigue valiendo como diagnóstico secundario: un pico ancho y sin
estructura a frecuencias que **escalen con el paso de integración** delata
castañeteo numérico del solver y no vibración física — y eso decidiría si "el
tracked vibra más" es una propiedad del vehículo o del integrador. Nota
relacionada: su acelerómetro en reposo ya marca σ 0.19–0.46 m/s², nueve a
veinte veces el ruido del sensor, por los 40 elementos discretos de cada oruga,
y **no se reproduce entre corridas ni con la semilla fija**.
