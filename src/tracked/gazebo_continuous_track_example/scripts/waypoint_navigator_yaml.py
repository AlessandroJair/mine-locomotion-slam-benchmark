#!/usr/bin/env python3

import rospy
import math
import yaml
import actionlib
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

class WaypointNavigatorYAML:
    def __init__(self):
        rospy.init_node('waypoint_navigator_yaml', anonymous=False)

        # Cargar configuracion desde archivo YAML
        config_file = rospy.get_param('~config_file', 'waypoints_config.yaml')
        self.load_config(config_file)

        # Override use_move_base desde parametro ROS (launch arg tiene prioridad)
        self.use_move_base = rospy.get_param('~use_move_base', self.use_move_base)

        self.current_waypoint_index = 0
        self.current_pose = None
        self.move_base_client = None
        self.map_ready = False
        self.map_update_count = 0

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/example_two_track/cmd_vel', Twist, queue_size=10)
        self.path_pub = rospy.Publisher('/waypoint_path', Path, queue_size=10)
        self.markers_pub = rospy.Publisher('/waypoint_markers', MarkerArray, queue_size=10, latch=True)

        # Subscribers
        odom_topic = rospy.get_param('~odom_topic', '/rtabmap/odom')
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        # skip_map_wait: for SLAM backends that don't build an occupancy
        # grid (e.g. ORB-SLAM3, which is sparse feature-based), there is
        # nothing to wait for -- move straight to navigation.
        self.skip_map_wait = rospy.get_param('~skip_map_wait', False)
        map_topic = rospy.get_param('~map_topic', '/rtabmap/grid_map')
        if not self.skip_map_wait:
            self.map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)

        # Inicializar move_base action client si se requiere
        if self.use_move_base:
            self._init_move_base_client()

        rospy.loginfo("Waypoint Navigator (YAML) - Tracked iniciado")
        rospy.loginfo(f"Total de waypoints: {len(self.waypoints)}")
        rospy.loginfo(f"Modo navegacion: {'move_base' if self.use_move_base else 'control simple'}")

        # Publicar visualizacion de waypoints
        self.publish_waypoint_markers()

    def load_config(self, config_file):
        """Carga la configuracion desde archivo YAML"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Cargar waypoints
            self.waypoints = []
            for wp in config['waypoints']:
                x = wp['x']
                y = wp['y']
                yaw_deg = wp['yaw']
                yaw_rad = math.radians(yaw_deg)
                name = wp.get('name', f"Waypoint {len(self.waypoints)}")

                self.waypoints.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'yaw': yaw_rad
                })

            # Cargar parametros de navegacion
            nav_params = config.get('navigation_params', {})
            self.goal_tolerance = nav_params.get('goal_tolerance', 0.5)
            self.angular_tolerance = nav_params.get('angular_tolerance', 0.1)
            self.max_linear_speed = nav_params.get('max_linear_speed', 0.5)
            self.max_angular_speed = nav_params.get('max_angular_speed', 0.5)
            self.linear_kp = nav_params.get('linear_kp', 0.5)
            self.angular_kp = nav_params.get('angular_kp', 1.0)
            self.use_move_base = nav_params.get('use_move_base', False)
            self.move_base_timeout = nav_params.get('move_base_timeout', 120.0)
            self.move_base_retry_count = nav_params.get('move_base_retry_count', 2)

            rospy.loginfo(f"Configuracion cargada desde: {config_file}")

        except Exception as e:
            rospy.logerr(f"Error cargando configuracion: {e}")
            rospy.signal_shutdown("Error en configuracion")

    def map_callback(self, msg):
        """Detecta cuando el mapa SLAM tiene datos de obstaculos reales para navegar"""
        self.map_update_count += 1
        if not self.map_ready:
            has_obstacles = any(c == 100 for c in msg.data)
            if (has_obstacles and self.map_update_count >= 3) or self.map_update_count >= 40:
                self.map_ready = True
                status = "con obstaculos" if has_obstacles else "SIN obstaculos (timeout)"
                rospy.loginfo(f"Mapa SLAM listo {status}: {self.map_update_count} actualizaciones. Iniciando navegacion.")

    def odom_callback(self, msg):
        """Callback para obtener la posicion actual del robot"""
        self.current_pose = msg.pose.pose

    def get_current_position(self):
        """Obtiene la posicion y orientacion actual"""
        if self.current_pose is None:
            return None, None, None

        x = self.current_pose.position.x
        y = self.current_pose.position.y

        orientation = self.current_pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, yaw = euler_from_quaternion(quaternion)

        return x, y, yaw

    def distance_to_goal(self, goal_x, goal_y):
        """Calcula la distancia al objetivo"""
        x, y, _ = self.get_current_position()
        if x is None:
            return float('inf')
        return math.sqrt((goal_x - x)**2 + (goal_y - y)**2)

    def angle_to_goal(self, goal_x, goal_y):
        """Calcula el angulo hacia el objetivo"""
        x, y, yaw = self.get_current_position()
        if x is None:
            return 0.0

        angle_to_target = math.atan2(goal_y - y, goal_x - x)
        angle_diff = angle_to_target - yaw

        # Normalizar entre -pi y pi
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        return angle_diff

    def publish_waypoint_markers(self):
        """Publica marcadores visuales de los waypoints en RViz"""
        marker_array = MarkerArray()

        for i, wp in enumerate(self.waypoints):
            # Marcador de posicion
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.pose.position.x = wp['x']
            marker.pose.position.y = wp['y']
            marker.pose.position.z = 0.5

            # Orientacion como quaternion
            marker.pose.orientation.z = math.sin(wp['yaw'] / 2)
            marker.pose.orientation.w = math.cos(wp['yaw'] / 2)

            marker.scale.x = 0.5
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

            # Marcador de texto
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "waypoint_labels"
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = wp['x']
            text_marker.pose.position.y = wp['y']
            text_marker.pose.position.z = 1.0

            text_marker.text = f"{i+1}: {wp['name']}"
            text_marker.scale.z = 0.3

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            marker_array.markers.append(text_marker)

        self.markers_pub.publish(marker_array)

    def _init_move_base_client(self):
        """Inicializa el action client de move_base con fallback"""
        rospy.loginfo("Conectando con move_base action server...")
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        connected = self.move_base_client.wait_for_server(rospy.Duration(30.0))
        if connected:
            rospy.loginfo("Conectado a move_base action server")
        else:
            rospy.logwarn("No se pudo conectar a move_base en 30s. Fallback a control simple.")
            self.move_base_client = None
            self.use_move_base = False

    def navigate_to_waypoint_move_base(self, waypoint):
        """Navega a un waypoint usando move_base con reintentos"""
        goal_x = waypoint['x']
        goal_y = waypoint['y']
        goal_yaw = waypoint['yaw']
        name = waypoint['name']

        for attempt in range(self.move_base_retry_count + 1):
            if rospy.is_shutdown():
                return False

            if attempt > 0:
                rospy.logwarn(f"Reintento {attempt}/{self.move_base_retry_count} para {name}")

            # Construir goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = goal_x
            goal.target_pose.pose.position.y = goal_y
            goal.target_pose.pose.position.z = 0.0

            q = quaternion_from_euler(0, 0, goal_yaw)
            goal.target_pose.pose.orientation = Quaternion(*q)

            rospy.loginfo(f"[move_base] Enviando goal: {name} ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_yaw):.1f} deg)")
            self.move_base_client.send_goal(goal)

            # Esperar resultado con timeout
            finished = self.move_base_client.wait_for_result(rospy.Duration(self.move_base_timeout))

            if not finished:
                rospy.logwarn(f"[move_base] Timeout ({self.move_base_timeout}s) alcanzado para {name}")
                self.move_base_client.cancel_goal()
                continue

            state = self.move_base_client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo(f"[move_base] {name} alcanzado exitosamente!")
                return True
            else:
                state_text = GoalStatus._to_str.get(state, str(state)) if hasattr(GoalStatus, '_to_str') else str(state)
                rospy.logwarn(f"[move_base] Fallo al alcanzar {name} (estado: {state_text})")

        rospy.logerr(f"[move_base] No se pudo alcanzar {name} despues de {self.move_base_retry_count + 1} intentos")
        return False

    def navigate_to_waypoint(self, waypoint):
        """Navegacion simple usando control proporcional"""
        goal_x = waypoint['x']
        goal_y = waypoint['y']
        goal_yaw = waypoint['yaw']
        name = waypoint['name']

        rate = rospy.Rate(10)  # 10 Hz
        timeout = self.move_base_timeout  # reutilizar timeout
        start_time = rospy.Time.now()
        stall_check_time = rospy.Time.now()
        last_distance = float('inf')
        stall_threshold = 0.1  # metros minimos de avance cada 10s

        rospy.loginfo(f"Navegando a: {name} ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_yaw):.2f})")

        while not rospy.is_shutdown():
            # Verificar timeout
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                rospy.logwarn(f"Timeout ({timeout}s) alcanzado navegando a {name}")
                break

            distance = self.distance_to_goal(goal_x, goal_y)

            # Verificar estancamiento cada 10 segundos
            stall_elapsed = (rospy.Time.now() - stall_check_time).to_sec()
            if stall_elapsed > 10.0:
                if abs(last_distance - distance) < stall_threshold:
                    rospy.logwarn(f"Robot estancado navegando a {name}. Abortando waypoint.")
                    break
                last_distance = distance
                stall_check_time = rospy.Time.now()

            # Verificar si llegamos al waypoint
            if distance < self.goal_tolerance:
                rospy.loginfo(f"{name} alcanzado! Distancia: {distance:.2f}m")

                # Ajustar orientacion final
                self.adjust_orientation(goal_yaw)
                break

            # Calcular velocidades
            angle_diff = self.angle_to_goal(goal_x, goal_y)

            cmd = Twist()

            # Velocidad angular minima para vencer friccion de orugas
            min_angular_speed = 0.35

            # Si el angulo es muy grande, primero girar
            if abs(angle_diff) > 0.3:  # ~17 grados
                cmd.linear.x = 0.0
                raw_angular = self.angular_kp * angle_diff
                clamped = max(-self.max_angular_speed,
                             min(self.max_angular_speed, raw_angular))
                # Aplicar velocidad minima para no atascarse
                if clamped > 0:
                    cmd.angular.z = max(min_angular_speed, clamped)
                else:
                    cmd.angular.z = min(-min_angular_speed, clamped)
            else:
                # Avanzar hacia el objetivo (velocidad minima para pendientes)
                linear_speed = self.linear_kp * distance
                min_speed = 0.35
                cmd.linear.x = max(min_speed, min(self.max_linear_speed, linear_speed))

                angular_speed = self.angular_kp * angle_diff
                cmd.angular.z = max(-self.max_angular_speed,
                                   min(self.max_angular_speed, angular_speed))

            self.cmd_vel_pub.publish(cmd)
            rate.sleep()

        # Detener el robot
        self.stop_robot()

    def adjust_orientation(self, goal_yaw):
        """Ajusta la orientacion final del robot"""
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        timeout = 15.0  # segundos maximo para ajustar orientacion
        settled_count = 0
        settled_threshold = 5  # frames consecutivos dentro de tolerancia

        while not rospy.is_shutdown():
            # Timeout de seguridad para evitar giros infinitos
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                rospy.logwarn("Timeout ajustando orientacion. Continuando.")
                break

            _, _, current_yaw = self.get_current_position()
            if current_yaw is None:
                continue

            yaw_diff = goal_yaw - current_yaw

            # Normalizar entre -pi y pi
            while yaw_diff > math.pi:
                yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi:
                yaw_diff += 2 * math.pi

            if abs(yaw_diff) < self.angular_tolerance:
                settled_count += 1
                if settled_count >= settled_threshold:
                    break
            else:
                settled_count = 0

            cmd = Twist()
            raw_angular = self.angular_kp * yaw_diff
            clamped = max(-self.max_angular_speed,
                         min(self.max_angular_speed, raw_angular))
            # Velocidad minima para vencer friccion de orugas
            min_angular_speed = 0.3
            if clamped > 0:
                cmd.angular.z = max(min_angular_speed, clamped)
            else:
                cmd.angular.z = min(-min_angular_speed, clamped)

            self.cmd_vel_pub.publish(cmd)
            rate.sleep()

        self.stop_robot()

    def stop_robot(self):
        """Detiene el robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        """Ejecuta la navegacion por todos los waypoints"""
        # Esperar a que el mapa SLAM este listo (min. 5 actualizaciones del grid map)
        if self.skip_map_wait:
            rospy.loginfo("skip_map_wait activo (SLAM sin grid map, ej. ORB-SLAM3). Iniciando navegacion directo.")
        else:
            rospy.loginfo("Esperando mapa SLAM antes de iniciar navegacion...")
            wait_start = rospy.Time.now()
            poll_rate = rospy.Rate(1.0)
            while not rospy.is_shutdown() and not self.map_ready:
                elapsed = (rospy.Time.now() - wait_start).to_sec()
                if elapsed > 90.0:
                    rospy.logwarn("Timeout 90s esperando mapa SLAM. Iniciando de todas formas.")
                    break
                rospy.loginfo_throttle(10.0, f"Esperando mapa SLAM... ({elapsed:.0f}s, actualizaciones={self.map_update_count})")
                poll_rate.sleep()
            rospy.sleep(3.0)  # Pausa para que el costmap estatico procese el mapa

        while not rospy.is_shutdown() and self.current_waypoint_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_index]

            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo(f"Progreso: {self.current_waypoint_index + 1}/{len(self.waypoints)}")

            if self.use_move_base and self.move_base_client is not None:
                success = self.navigate_to_waypoint_move_base(waypoint)
                if not success:
                    rospy.logwarn(f"move_base fallo. Usando control simple para este waypoint.")
                    self.navigate_to_waypoint(waypoint)
            else:
                self.navigate_to_waypoint(waypoint)

            self.current_waypoint_index += 1
            rospy.sleep(2.0)  # Pausa entre waypoints

        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("MISION COMPLETADA! Todos los waypoints alcanzados")
        rospy.loginfo("="*60)
        self.stop_robot()

if __name__ == '__main__':
    try:
        navigator = WaypointNavigatorYAML()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
