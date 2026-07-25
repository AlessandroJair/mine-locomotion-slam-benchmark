#!/usr/bin/env python3

import rospy
import yaml
import math
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty, EmptyResponse

class WaypointRecorder:
    """
    Graba waypoints interactivamente mientras conduces el robot manualmente.

    Uso:
    1. Lanza este nodo
    2. Conduce el robot manualmente a cada posicion deseada
    3. Llama al servicio para grabar cada waypoint:
       rosservice call /record_waypoint
    4. Al finalizar, llama al servicio para guardar:
       rosservice call /save_waypoints
    """

    def __init__(self):
        rospy.init_node('waypoint_recorder', anonymous=False)

        self.waypoints = []
        self.current_pose = None
        self.output_file = rospy.get_param('~output_file', 'recorded_waypoints.yaml')

        # Subscribers
        self.odom_sub = rospy.Subscriber('/rtabmap/odom', Odometry, self.odom_callback)

        # Services
        self.record_srv = rospy.Service('/record_waypoint', Empty, self.record_waypoint)
        self.save_srv = rospy.Service('/save_waypoints', Empty, self.save_waypoints)
        self.clear_srv = rospy.Service('/clear_waypoints', Empty, self.clear_waypoints)

        rospy.loginfo("="*60)
        rospy.loginfo("Waypoint Recorder - Husky iniciado")
        rospy.loginfo("="*60)
        rospy.loginfo("Servicios disponibles:")
        rospy.loginfo("  - /record_waypoint : Graba la posicion actual")
        rospy.loginfo("  - /save_waypoints  : Guarda waypoints a archivo")
        rospy.loginfo("  - /clear_waypoints : Borra todos los waypoints")
        rospy.loginfo(f"Archivo de salida: {self.output_file}")
        rospy.loginfo("="*60)

    def odom_callback(self, msg):
        """Callback para obtener la posicion actual"""
        self.current_pose = msg.pose.pose

    def get_current_position(self):
        """Obtiene posicion y orientacion actuales"""
        if self.current_pose is None:
            return None, None, None

        x = self.current_pose.position.x
        y = self.current_pose.position.y

        orientation = self.current_pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, yaw = euler_from_quaternion(quaternion)

        return x, y, yaw

    def record_waypoint(self, req):
        """Servicio para grabar un waypoint"""
        x, y, yaw = self.get_current_position()

        if x is None:
            rospy.logwarn("No hay datos de odometria disponibles")
            return EmptyResponse()

        waypoint_num = len(self.waypoints) + 1
        name = f"Waypoint {waypoint_num}"

        waypoint = {
            'name': name,
            'x': float(x),
            'y': float(y),
            'yaw': float(math.degrees(yaw))
        }

        self.waypoints.append(waypoint)

        rospy.loginfo("="*60)
        rospy.loginfo(f"Waypoint {waypoint_num} grabado:")
        rospy.loginfo(f"  Posicion: ({x:.3f}, {y:.3f})")
        rospy.loginfo(f"  Orientacion: {math.degrees(yaw):.2f}")
        rospy.loginfo(f"  Total de waypoints: {len(self.waypoints)}")
        rospy.loginfo("="*60)

        return EmptyResponse()

    def save_waypoints(self, req):
        """Servicio para guardar waypoints a archivo YAML"""
        if not self.waypoints:
            rospy.logwarn("No hay waypoints para guardar")
            return EmptyResponse()

        config = {
            'waypoints': self.waypoints,
            'navigation_params': {
                'goal_tolerance': 0.5,
                'angular_tolerance': 0.1,
                'max_linear_speed': 0.5,
                'max_angular_speed': 0.5,
                'linear_kp': 0.5,
                'angular_kp': 1.0,
                'use_move_base': False
            }
        }

        try:
            with open(self.output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            rospy.loginfo("="*60)
            rospy.loginfo(f"{len(self.waypoints)} waypoints guardados en:")
            rospy.loginfo(f"  {self.output_file}")
            rospy.loginfo("="*60)

            # Mostrar resumen
            rospy.loginfo("\nResumen de waypoints:")
            for i, wp in enumerate(self.waypoints):
                rospy.loginfo(f"  {i+1}. {wp['name']}: ({wp['x']:.2f}, {wp['y']:.2f}, {wp['yaw']:.1f})")

        except Exception as e:
            rospy.logerr(f"Error guardando waypoints: {e}")

        return EmptyResponse()

    def clear_waypoints(self, req):
        """Servicio para borrar todos los waypoints"""
        count = len(self.waypoints)
        self.waypoints = []
        rospy.loginfo(f"{count} waypoints borrados")
        return EmptyResponse()

    def run(self):
        """Mantiene el nodo activo"""
        rospy.spin()

if __name__ == '__main__':
    try:
        recorder = WaypointRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass
