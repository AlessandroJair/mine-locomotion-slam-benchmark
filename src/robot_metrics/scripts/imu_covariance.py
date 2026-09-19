#!/usr/bin/env python3
"""EN DESUSO desde el 2026-09-09.  Lo sustituye imu_ahrs.py, que ademas
de las covarianzas FUSIONA la orientacion en vez de dejar pasar la del
plugin, que es la pose exacta de Gazebo.  Ningun launch lo lanza ya.
Se conserva porque documenta por que las covarianzas venian en cero.

Rellena las covarianzas del IMU, que el plugin de Gazebo deja en cero.

POR QUE HACE FALTA UN NODO
--------------------------
gazebo_ros_imu_sensor ata el ruido y la covarianza al MISMO parametro:

    const double gn2 = gaussian_noise * gaussian_noise;
    imu_msg.orientation_covariance[0]       = gn2;
    imu_msg.angular_velocity_covariance[0]  = gn2;
    imu_msg.linear_acceleration_covariance[0] = gn2;

Este proyecto necesita que TODO el ruido salga de un generador sembrado, para
que `gzserver --seed N` haga la corrida reproducible.  El generador propio del
plugin es el rand() de C, que la semilla no alcanza, asi que el ruido se movio
a los bloques <noise> del <sensor> (esos si usan ignition::math::Rand) y el
<gaussianNoise> del plugin quedo en 0.0.  Efecto colateral: las tres matrices
de covarianza quedaron en cero, y segun la spec de sensor_msgs/Imu una matriz
toda en cero significa "covarianza desconocida".  RTAB-Map consume este topic
(Imu/Enable: true, wait_imu_to_init: true).

No se puede tener las dos cosas con el plugin tal cual, de ahi este nodo: el
plugin publica crudo en ~input y esto republica en ~output con las diagonales
puestas.  Los consumidores (rtabmap, metrics_logger, rearing_guard, wheel_odom)
siguen leyendo /imu/data sin enterarse.

LOS VALORES
-----------
Varianza = stddev^2, con los stddev declarados en sim_config.yaml, que son los
que los bloques <noise> aplican de verdad (verificado midiendo 250 muestras:
0.00956 / 0.00876 / 0.00820 contra 0.009 declarado).

La orientacion es aparte: los bloques <noise> del <imu> perturban velocidad
angular y aceleracion, NO la orientacion, que sale de la pose exacta de Gazebo.
O sea es verdad de terreno sin ruido.  Declarar varianza cero haria que un
filtro le diera confianza infinita, asi que se expone como parametro con un
valor chico por defecto.  Es una decision de modelado, no una propiedad medida,
y como tal deberia declararse en el paper.
"""
import rospy
from sensor_msgs.msg import Imu


class ImuCovariance(object):
    def __init__(self):
        # stddev, no varianza: se declaran igual que en sim_config.yaml
        w_sd = float(rospy.get_param('~angular_velocity_stddev', 0.009))
        a_sd = float(rospy.get_param('~linear_acceleration_stddev', 0.021))
        o_sd = float(rospy.get_param('~orientation_stddev', 0.01))

        self.w_var = w_sd * w_sd
        self.a_var = a_sd * a_sd
        self.o_var = o_sd * o_sd

        self.pub = rospy.Publisher('~output', Imu, queue_size=10)
        rospy.Subscriber('~input', Imu, self.cb, queue_size=10)

        rospy.loginfo('imu_covariance: rellenando diagonales')
        rospy.loginfo('  velocidad angular  stddev %.4f -> var %.3e', w_sd, self.w_var)
        rospy.loginfo('  aceleracion lineal stddev %.4f -> var %.3e', a_sd, self.a_var)
        rospy.loginfo('  orientacion        stddev %.4f -> var %.3e  (valor de modelado)',
                      o_sd, self.o_var)

    def cb(self, msg):
        # Solo se tocan las covarianzas; la medicion pasa intacta, asi que el
        # ruido sigue siendo el que genero el generador sembrado.
        msg.orientation_covariance = [self.o_var, 0.0, 0.0,
                                      0.0, self.o_var, 0.0,
                                      0.0, 0.0, self.o_var]
        msg.angular_velocity_covariance = [self.w_var, 0.0, 0.0,
                                           0.0, self.w_var, 0.0,
                                           0.0, 0.0, self.w_var]
        msg.linear_acceleration_covariance = [self.a_var, 0.0, 0.0,
                                              0.0, self.a_var, 0.0,
                                              0.0, 0.0, self.a_var]
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('imu_covariance')
    ImuCovariance()
    rospy.spin()
