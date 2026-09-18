# El Rocker-Bogie pierde el apoyo de las ruedas

Investigación del 2026-09-04/05. Todos los números de este documento están
medidos; los que no pude medir se dicen como tales.

---

## El síntoma

Sobre la ruta de la mina el Rocker-Bogie circula **hasta un 20 % del tiempo con
menos de seis ruedas apoyadas**, y en las maniobras de giro baja a dos. Al
observarlo en el visor: se desliza pendiente abajo, no consigue girar limpio y
se queda escorado.

Importa porque es la plataforma cuya ventaja declarada es justamente mantener
las seis ruedas en contacto. El README del repositorio publica al rocker
ganando en todo:

| | Husky | Tracked | Rocker |
|---|---:|---:|---:|
| Aceleración vertical máx. (m/s²) | 13.12 | 16.85 | **7.68** |
| Ángulo de balanceo máx. (rad) | 0.577 | 0.503 | **0.459** |
| ATE (m) | 1.46 | 1.76 | **0.72** |

Hoy pierde en todas. El pico vertical pasó de 7.68 a **159.25 m/s²** — veinte
veces peor — y de ser el mejor de los tres al peor.

---

## Cómo es el mecanismo, de verdad

Verificado recorriendo la cadena cinemática rueda por rueda sobre el fichero
que Gazebo spawnea:

```
por lado:  rocker_pivot_left/right   base_link -> balancín     ±0.6 rad (±34.4°)
           rev_3 / rev_4             balancín  -> bogie         ±0.6 rad
           rev_5..rev_8              4 juntas de dirección      ±1.1 rad
           rev_9..rev_14             6 ruedas                   continuous
```

Es un rocker-bogie correcto: balancín **más** bogie por lado, con las tres
ruedas de cada lado colgando del mismo balancín. Las juntas de suspensión no
llevan controlador ni amortiguamiento apreciable — ruedan libres.

**El diferencial no está activo.** Hay dos implementaciones y las dos están
apagadas:

```
launch:  <arg name="differential_enabled" default="false" />   -> quita la junta gearbox
nodo:    <param name="enabled" value="false" />                -> el diferencial software
```

El propio launch lo llama *"a known-incomplete state, not a preference"*.

**Los cuatro ángulos de suspensión SÍ se publican** en
`/rocker_bogie/joint_states`, pero `metrics_logger` no los guarda en
`metrics.csv`. Son cuatro columnas que faltan y que habrían acortado mucho esta
investigación.

---

## Qué se ha descartado, con medida

### 1. El diferencial ausente — NO es la causa

Se activó el diferencial software (`rocker_differential.py`, par de
acoplamiento en vez de restricción rígida) y se corrió una vuelta completa:

| | sin diferencial | con diferencial soft | cambio |
|---|---:|---:|---:|
| `az_max` [m/s²] | 159.25 | 29.34 | **−81.6 %** |
| vibración RMS | 3.554 | 1.920 | **−46.0 %** |
| ATE optimizado [m] | 1.219 | 0.267 | **−78.1 %** |
| tiempo con <6 ruedas | 19.7 % | 16.1 % | −18 % |
| `Rueda_4_1` al aire | 6.4 % | 5.9 % | −7 % |

**El diferencial quita el 80 % de los impactos y no toca el levantamiento.** El
16.1 % cae dentro de la dispersión entre corridas sin diferencial (19.7 % y
14.4 %). Lo que hace el acoplamiento es impedir que cada pérdida de apoyo se
descargue de golpe, no impedir la pérdida.

La versión rígida (junta `gearbox`) **no es utilizable**: parte quieta a
z = 1.095 y a los 3.0 s está en z = 2.0 volcando. Y el apaño documentado en el
launch (`spawn_z:=0.75`) es una medida de banco plano — en la galería el suelo
del punto de spawn deja el robot posado a **1.0931 m**, así que 0.75 lo entierra
34 cm y ODE lo expulsa. Barrer `gearbox_cfm` tres décadas tampoco lo arregla
(ya estaba probado y anotado en el launch).

### 2. El amortiguamiento — cambia contacto por absorción, no lo arregla

El original de GitHub (2026-03-12) llevaba las juntas mucho más amortiguadas:

| junta | ORIGINAL | HOY | factor |
|---|---|---|---:|
| `rocker_pivot_*` | damping 80, friction 4 | 0.5 / 0.2 | ÷160, ÷20 |
| `rev_3`, `rev_4` | damping 80, friction 10 | 0.5 / 0.2 | ÷160, ÷50 |
| `differential_joint` | damping 80, friction 6 | **no existe** | — |

Restaurando damping 80 sobre la ruta completa:

| | damping 0.5 | damping 80 | cambio |
|---|---:|---:|---:|
| vibración RMS | 2.281 | 1.693 | −26 % |
| `az_max` | 160.9 | 53.3 | −67 % |
| pico \|a\| | 365 | 123 | −66 % |
| **tiempo con <6 ruedas** | **19.7 %** | **51.7 %** | **+162 %** |
| **`Rueda_4_1` al aire** | **6.4 %** | **38.0 %** | **+496 %** |

**El amortiguamiento absorbe los golpes y empeora el apoyo**: con 80 el
mecanismo ya no puede seguir al terreno y pasa la mitad del recorrido con
ruedas al aire.

### 3. El recorrido de los topes — NO es el cuello de botella

Hipótesis: en el giro `rev_3` barría 65.2° de los 68.8° disponibles, o sea
golpeaba los dos topes. Se ampliaron a ±1.2 rad (137.5° de recorrido):

| fase | topes ±0.6 | topes ±1.2 |
|---|---:|---:|
| ruedas en el giro | 3.93 | **3.78** |
| `rev_4` barrido | 38.8° | **72.4°** |
| deriva lateral (arco) | +0.998 m | +0.999 m |
| ruedas en el arco | 5.59 | 5.60 |

**El bogie usa todo el recorrido nuevo —72°, más que los 68.8° que tenía en
total— y el apoyo no mejora.** El arco sale idéntico porque ahí nunca llegaba a
tocar tope, lo que confirma que el cambio hizo exactamente lo que decía.

### 4. La magnitud del par de las ruedas — NO escala

Hipótesis: el par motor reacciona sobre el brazo y lo levanta. Se repitió el
mismo giro a cuatro velocidades sobre la misma rampa cruzada:

| w pedida | ruedas | `rev_3` | `rev_4` | w real [°/s] |
|---:|---:|---:|---:|---:|
| 0.000 (reposo) | 6.00 | 0.0° | 0.0° | 0.0 |
| 0.125 | 3.87 | 15.1° | **33.2°** | 3.0 |
| 0.250 | 4.40 | 7.6° | **38.4°** | 13.6 |
| 0.375 | 4.00 | 5.9° | **37.9°** | 19.1 |
| 0.500 | 4.38 | 4.5° | **34.4°** | 16.7 |

**El enrollamiento es constante a todas las velocidades**, incluida la más
lenta, que solo consigue 3 °/s. Si lo produjera la magnitud del par, girando
seis veces más despacio tendría que caer.

**Por qué no escala:** la fuerza que enrolla el brazo la fija el *rozamiento*,
no la velocidad. Con las ruedas dirigidas a ±55° están arrastrando de lado, y
una rueda que patina transmite `μ·N` independientemente de lo deprisa que
patine.

---

## Lo que sí está establecido

### En estático el mecanismo funciona

Banco de rampa cruzada: media vía sobre un plano inclinado 20° y media sobre
llano (`world/banco/rampa_cruzada.world`).

```
asentado, quieto sobre la arista
  ruedas en contacto: 6.00 de 6
  Rueda_1_1  100.5 N (llano)   Rueda_2_1   70.1 N (rampa)
  Rueda_3_1   87.0 N (llano)   Rueda_4_1   68.4 N (rampa)
  Rueda_5_1   77.0 N (llano)   Rueda_6_1   45.8 N (rampa)
  roll -7.5°   suspension 1-2°   desequilibrio de carga 2.2:1
```

Tres ruedas en cada superficie, **las seis apoyadas**, y el balanceo del cuerpo
(−7.5°) es exactamente el que pide la geometría: 0.117 m de desnivel sobre
0.79 m de vía = 8.4°.

Los pivotes de balancín y bogie giran sobre el eje **lateral**, así que absorben
diferencias de *cabeceo*, no de *balanceo*. Una pendiente cruzada la toma el
cuerpo rodando, y la toma bien.

### Es el giro lo que rompe el apoyo

| fase (misma rampa) | ruedas | carga máx/mín | deriva lateral | `rev_3` | `rev_4` |
|---|---:|---:|---:|---:|---:|
| asentado, quieto | 6.00 | 2.2:1 | — | 0.0° | 0.0° |
| avanzando recto 3.6 m | 6.00 | — | **−0.299 m** | 2.4° | 1.1° |
| arco (v 0.25, w 0.35) | 5.59 | 4.6:1 | **+0.998 m** | 4.8° | 24.1° |
| giro en el sitio (w 0.5) | **3.93** | **11.1:1** | −0.060 m | **65.2°** | 38.8° |
| parado, después | 5.65 | 10.1:1 | — | 3.4° | 1.8° |

Y el balanceo al terminar es **−17.5°**, contra los −7.5° que tenía bien montado
a caballo de la arista: **después de girar no recupera la postura**.

Avanzando en línea recta ya se va 30 cm de lado en 3.6 m (8 %), con el rumbo
casi sin cambiar — se traduce lateralmente pendiente abajo.

### Se queda tensionado, y es biestable

Fases de reposo **entre** giros consecutivos:

```
  reposo tras w=0.125    4.00 ruedas   carga máx/mín  2363
  reposo tras w=0.250    6.00 ruedas   carga máx/mín     8
  reposo tras w=0.375    4.55 ruedas   carga máx/mín  2173
  reposo tras w=0.500    6.00 ruedas   carga máx/mín     2
```

Tras girar, a veces se queda apoyado en cuatro ruedas con una tocando sin carga
apreciable, y a veces se recupera del todo. No es un transitorio que se asiente:
es una postura estable en la que se queda hasta que algo lo saca.

### Hay un punto de la ruta que lo tumba siempre

Entre los metros 108 y 124 la ruta **baja 2.9 m en 16 m** (~18°) y en el metro
115 la pendiente se rompe: el cabeceo salta de −18° a −10° en dos metros.

| m | z | pitch | ruedas (A) | ruedas (B) | ruedas (dirfix) |
|---:|---:|---:|---:|---:|---:|
| 112 | 2.43 | −17.0° | 5.50 | 5.97 | 5.52 |
| 114 | 1.87 | −18.3° | 4.81 | 5.61 | 4.50 |
| **116** | 1.54 | −9.6° | **4.57** | **4.29** | **2.64** |
| 118 | 1.25 | −10.6° | 5.00 | 5.46 | 5.21 |

**Ocurre en todas las corridas.** El rocker corona la arista apoyado en un
extremo. Con esa actitud el error de rumbo supera `align_threshold` y el
seguidor pide un giro sobre el sitio; girar apoyado en dos ruedas a −23° es lo
que produjo un pico de **2229 m/s² (227 g)**, que es el solver reventando, no un
choque físico.

---

## Lo que queda apuntando: el arrastre lateral

Descartados el diferencial, el amortiguamiento, el recorrido y la magnitud del
par, la fuerza que enrolla el bogie es el **rozamiento lateral de las ruedas
dirigidas**: `mu2 = 0.75` con las ruedas a ±55°.

**No se ha podido probar** porque `mu1`/`mu2` y los parámetros de contacto están
declarados en `sim_config.yaml` y verificados por el chequeo 16 de
`check_sim_parity` en las tres plataformas — se estandarizaron a propósito para
que las tres rueden sobre el mismo contacto. Cambiarlos solo en el rocker
rompería la comparación.

La prueba que lo cerraría: correr el banco de rampa cruzada con `mu2` alterado
**solo en el banco**, no en la campaña. Si el enrollamiento cae, es el arrastre
lateral, y entonces es una **limitación conocida del modelo que hay que
reportar**, no un bug que se arregla por un lado.

### Y una diferencia con el original que no se ha investigado

El contacto de las ruedas también cambió respecto al GitHub de marzo:

| | ORIGINAL | HOY | factor |
|---|---:|---:|---:|
| `mu1` | 100.0 | 1.0 | ÷100 |
| `kd` | 100.0 | 1.0 | ÷100 |
| `maxVel` | 0.01 | 1.0 | ×100 |
| `minDepth` | 0.001 | 0.003 | ×3 |

`maxVel` es la velocidad con la que ODE expulsa una penetración: pasó de 1 cm/s
a **1 m/s**. Con `kd` cien veces menor, el contacto pasó de rígido-y-suave a
blando-y-violento, y eso por sí solo explicaría los picos de cientos de m/s².
Estos valores **también están protegidos por la paridad**.

---

## Cómo reproducirlo

| script | qué hace |
|---|---|
| `~/rampa_cruzada.world` → `world/banco/` | media vía a 20°, media en llano |
| `~/banco_rampa_nodo.py` | secuencia asentar / avanzar / arco / giro, mide contacto, carga, suspensión y deriva |
| `~/banco_rampa.sh` | monta el mundo, lanza y **restaura el launch** |
| `~/banco_impulso_nodo.py` | el mismo giro a cuatro velocidades: prueba de escalado con el par |
| `~/amplia_topes.py` | ±0.6 → ±1.2 rad en los cuatro pivotes, con `--restaura` |
| `~/dos_ruedas.py`, `~/punto_115m.py` | contacto por régimen y perfil del metro 115 |

Los benches parchean el launch y lo restauran solos. Si alguno se corta a la
brava, `~/restaura_rampa.sh` y `~/restaura_xacro.sh` devuelven el repo.

## Lo primero que haría al retomarlo

1. **Registrar los cuatro ángulos de suspensión en `metrics.csv`.** Existen en
   `/rocker_bogie/joint_states` y no se guardan; sin ellos, todo lo de arriba ha
   costado benches a medida.
2. La prueba de `mu2` en el banco, para cerrar o descartar el arrastre lateral.
3. Decidir qué hacer con el diferencial: quita el 80 % de los impactos y el
   78 % del ATE optimizado por dos parámetros. Está apagado por un vuelco de
   spawn cuyo apaño documentado estaba mal medido.
