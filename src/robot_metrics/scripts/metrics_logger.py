#!/usr/bin/env python3
"""Per-run logger for the three-platform mine-exploration comparison.

WHAT CHANGED FOR THE REVISION
=============================
The previous version wrote a single <robot>.csv that was overwritten by every
run, so only the last of the three repetitions survived and no per-run spread
could be reported.  It also kept no raw ground-truth trajectory, no record of
tracking losses, and no wheel-contact data.

Each run now gets its own directory

    <output_dir>/<robot_name>/run<NN>/
        metrics.csv          time series, one row per 50 Hz sample
        gt_traj.tum          raw ground-truth poses, TUM format
        est_traj.tum         SLAM pose estimates, TUM format
        events.csv           tracking losses, relocalizations, loop closures
        columns.csv          data dictionary: column, unit, description
        run_meta.yaml        provenance for this run

so the three repetitions coexist and aggregate_runs.py can report mean +/- std.

GROUND TRUTH - PROVENANCE
=========================
Ground truth is the pose of the robot model as integrated by the physics
engine, read from /gazebo/model_states (gazebo_msgs/ModelStates).

  * It is the exact state vector.  Gazebo does not filter, smooth or add noise
    to it; there is no estimator between ODE and this topic.
  * It refers to the model's canonical link (base_link), the same frame the
    SLAM estimate is expressed in, so ATE and RPE need no extrinsic
    compensation.  Sensor mounting offsets are documented separately by
    extract_sensor_poses.py.
  * ModelStates carries NO header stamp.  Every sample is therefore stamped
    with the ROS sim clock at reception (use_sim_time:=true).  The topic is
    published on the Gazebo update loop at ~1 kHz while this node logs at
    50 Hz, so the stamping error is bounded by one physics step, 1 ms - three
    orders of magnitude below the ATE values of interest.
  * The unmodified poses are written to gt_traj.tum.  The columns in
    metrics.csv are the same numbers, not a processed version of them.

This is also stated in robot_metrics/config/sim_config.yaml under
`ground_truth`, which is the version the paper should cite.
"""

import math
import os
import subprocess
import csv as csvmod

import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

from metrics_io import (METRIC_COLUMNS, contact_columns, csv_header,
                        write_tum, write_units_reference)

try:
    from gazebo_msgs.msg import ModelStates
    HAS_GAZEBO_MSGS = True
except ImportError:
    HAS_GAZEBO_MSGS = False

try:
    from gazebo_msgs.msg import ContactsState
    HAS_CONTACT_MSGS = True
except ImportError:
    HAS_CONTACT_MSGS = False

try:
    from rtabmap_msgs.msg import Info as RtabmapInfo
    from rtabmap_msgs.msg import OdomInfo
    HAS_RTABMAP_MSGS = True
except ImportError:
    try:
        from rtabmap_ros.msg import Info as RtabmapInfo
        from rtabmap_ros.msg import OdomInfo
        HAS_RTABMAP_MSGS = True
    except ImportError:
        HAS_RTABMAP_MSGS = False

# A pose covariance at or above this value is RTAB-Map's way of publishing
# "I do not know where I am".
LOST_COVARIANCE = 9999.0


class MetricsLogger(object):

    def __init__(self):
        rospy.init_node('metrics_logger', anonymous=False)

        self.robot_name = rospy.get_param('~robot_name', 'robot')
        self.run_id = int(rospy.get_param('~run_id', 1))
        self.model_name = rospy.get_param('~model_name', self.robot_name)
        cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        odom_topic = rospy.get_param('~odom_topic', '/rtabmap/odom')
        base_output = rospy.get_param(
            '~output_dir', os.path.expanduser('~/metrics_output'))
        self.tracking_loss_timeout = float(
            rospy.get_param('~tracking_loss_timeout', 1.0))
        # Two GT poses closer than this are considered the same place, which
        # is how a reported loop closure is judged true or false.
        self.loop_true_radius = float(
            rospy.get_param('~loop_true_radius', 3.0))

        wheel_links = rospy.get_param('~wheel_links', [])
        if isinstance(wheel_links, str):
            wheel_links = [w.strip() for w in wheel_links.split(',') if w.strip()]
        self.wheel_links = wheel_links

        self.run_dir = os.path.join(
            base_output, self.robot_name, 'run{:02d}'.format(self.run_id))
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.columns = list(METRIC_COLUMNS) + contact_columns(self.wheel_links)

        self.rows = []
        self.gt_poses = []      # (t, x, y, z, qx, qy, qz, qw)
        self.est_poses = []
        self.events = []        # (t, type, detail, gt_x, gt_y, gt_distance)

        self._init_state()
        self._init_subscribers(odom_topic, cmd_vel_topic)

        self.rate = rospy.Rate(50)
        rospy.on_shutdown(self.save)

        rospy.loginfo('MetricsLogger: robot=%s run=%d', self.robot_name,
                      self.run_id)
        rospy.loginfo('  output    : %s', self.run_dir)
        rospy.loginfo('  odom      : %s', odom_topic)
        rospy.loginfo('  cmd_vel   : %s', cmd_vel_topic)
        rospy.loginfo('  gt model  : %s (via /gazebo/model_states, unfiltered)',
                      self.model_name)
        if self.wheel_links:
            rospy.loginfo('  contacts  : %d wheels', len(self.wheel_links))

    # ------------------------------------------------------------------
    def _init_state(self):
        self.pitch = self.roll = self.yaw = 0.0
        self.accel = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]
        self.ang_acc_x = self.ang_acc_z = 0.0
        self._prev_wx = self._prev_wz = None
        self._prev_imu_t = None

        self.odom = [0.0] * 3
        self.odom_rpy = [0.0] * 3
        self.odom_vx = self.odom_vyaw = 0.0
        self.cmd_vx = self.cmd_vyaw = 0.0

        self.gt = [0.0] * 3
        self.gt_rpy = [0.0] * 3
        self.gt_v = [0.0, 0.0]
        self.gt_quat = [0.0, 0.0, 0.0, 1.0]
        self.gt_distance = 0.0
        self._prev_gt_xy = None
        self.gt_ready = False

        self.loop_closures = 0
        self.proximity = 0
        self.inliers = self.matches = 0
        self.icp_ratio = 0.0
        self.icp_corr = 0
        self.proc_time = 0.0

        self.tracking_lost = 0
        self.tracking_loss_count = 0
        self.relocalization_count = 0
        self._last_odom_time = None

        # node id -> ground-truth position, used to classify loop closures
        self._node_gt = {}

        self.contact = {l: 0 for l in self.wheel_links}
        self.normal_force = {l: 0.0 for l in self.wheel_links}

        self.tf_dx = self.tf_dy = self.tf_dyaw = 0.0
        self._prev_tf = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def _init_subscribers(self, odom_topic, cmd_vel_topic):
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        rospy.Subscriber(cmd_vel_topic, Twist, self.cmd_cb)

        if HAS_GAZEBO_MSGS:
            rospy.Subscriber('/gazebo/model_states', ModelStates,
                             self.model_states_cb, queue_size=1)
        else:
            rospy.logerr('gazebo_msgs missing: NO GROUND TRUTH will be '
                         'recorded and ATE/RPE cannot be computed.')

        if HAS_CONTACT_MSGS:
            for link in self.wheel_links:
                rospy.Subscriber('/wheel_contacts/{}'.format(link),
                                 ContactsState, self.contact_cb,
                                 callback_args=link, queue_size=1)
        elif self.wheel_links:
            rospy.logwarn('gazebo_msgs.ContactsState unavailable: wheel '
                          'contact columns will stay at zero.')

        if HAS_RTABMAP_MSGS and 'rtabmap' in odom_topic:
            rospy.Subscriber('/rtabmap/info', RtabmapInfo, self.info_cb)
            rospy.Subscriber('/rtabmap/odom_info', OdomInfo, self.odom_info_cb)

    # ---------------- callbacks ----------------
    def imu_cb(self, msg):
        q = msg.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w])
        self.accel = [msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z]
        wx, wy, wz = (msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z)
        t = msg.header.stamp.to_sec()
        if self._prev_imu_t is not None:
            dt = t - self._prev_imu_t
            if dt > 0:
                self.ang_acc_x = (wx - self._prev_wx) / dt
                self.ang_acc_z = (wz - self._prev_wz) / dt
        self._prev_wx, self._prev_wz, self._prev_imu_t = wx, wz, t
        self.ang_vel = [wx, wy, wz]

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom = [p.x, p.y, p.z]
        self.odom_rpy = list(euler_from_quaternion([q.x, q.y, q.z, q.w]))
        self.odom_vx = msg.twist.twist.linear.x
        self.odom_vyaw = msg.twist.twist.angular.z
        self._last_odom_time = rospy.get_time()

        # RTAB-Map signals a lost track with a saturated covariance.
        lost = msg.pose.covariance[0] >= LOST_COVARIANCE
        self._set_tracking(lost, 'covariance')

        self.est_poses.append((rospy.get_time(), p.x, p.y, p.z,
                               q.x, q.y, q.z, q.w))

    def cmd_cb(self, msg):
        self.cmd_vx = msg.linear.x
        self.cmd_vyaw = msg.angular.z

    def model_states_cb(self, msg):
        try:
            i = msg.name.index(self.model_name)
        except ValueError:
            rospy.logwarn_throttle(
                10.0, 'model "%s" not in /gazebo/model_states; names=%s',
                self.model_name, list(msg.name))
            return

        pose = msg.pose[i]
        tw = msg.twist[i]
        q = pose.orientation
        self.gt = [pose.position.x, pose.position.y, pose.position.z]
        self.gt_quat = [q.x, q.y, q.z, q.w]
        self.gt_rpy = list(euler_from_quaternion([q.x, q.y, q.z, q.w]))
        self.gt_v = [tw.linear.x, tw.linear.y]

        if self._prev_gt_xy is not None:
            self.gt_distance += math.hypot(self.gt[0] - self._prev_gt_xy[0],
                                           self.gt[1] - self._prev_gt_xy[1])
        self._prev_gt_xy = (self.gt[0], self.gt[1])
        self.gt_ready = True

    def contact_cb(self, msg, link):
        if not msg.states:
            self.contact[link] = 0
            self.normal_force[link] = 0.0
            return
        self.contact[link] = 1
        total = 0.0
        for st in msg.states:
            f = st.total_wrench.force
            total += math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
        self.normal_force[link] = total

    def info_cb(self, msg):
        # Remember where each map node was created, so a later loop closure
        # onto that node can be judged against ground truth.
        if getattr(msg, 'refId', 0) > 0 and self.gt_ready:
            self._node_gt[msg.refId] = (self.gt[0], self.gt[1])

        lc_id = getattr(msg, 'loopClosureId', 0)
        if lc_id > 0:
            self.loop_closures += 1
            self._classify_loop_closure(lc_id)

        if getattr(msg, 'proximityDetectionId', 0) > 0:
            self.proximity += 1

    def _classify_loop_closure(self, lc_id):
        """True positive if the two linked nodes really are the same place."""
        ref = self._node_gt.get(lc_id)
        if ref is None or not self.gt_ready:
            verdict, dist = 'unknown', float('nan')
        else:
            dist = math.hypot(self.gt[0] - ref[0], self.gt[1] - ref[1])
            verdict = 'true' if dist <= self.loop_true_radius else 'false'
        self.log_event('loop_closure',
                       'id={} verdict={} gt_gap={:.3f}m'.format(
                           lc_id, verdict, dist))

    def odom_info_cb(self, msg):
        self.inliers = msg.inliers
        self.matches = msg.matches
        self.icp_ratio = getattr(msg, 'icpInliersRatio', 0.0)
        self.icp_corr = getattr(msg, 'icpCorrespondences', 0)
        self.proc_time = getattr(msg, 'timeEstimation', 0.0)
        if hasattr(msg, 'lost'):
            self._set_tracking(bool(msg.lost), 'odom_info')

    # ---------------- tracking-loss bookkeeping ----------------
    def _set_tracking(self, lost, source):
        if lost and not self.tracking_lost:
            self.tracking_lost = 1
            self.tracking_loss_count += 1
            self.log_event('tracking_loss', 'source={}'.format(source))
        elif not lost and self.tracking_lost:
            self.tracking_lost = 0
            self.relocalization_count += 1
            self.log_event('relocalization', 'source={}'.format(source))

    def log_event(self, kind, detail):
        """Record an event with where on the route it happened."""
        self.events.append((rospy.get_time(), kind, detail,
                            self.gt[0], self.gt[1], self.gt_distance))
        rospy.loginfo('[event] %-15s %s  at gt=(%.2f, %.2f) s=%.1fm',
                      kind, detail, self.gt[0], self.gt[1], self.gt_distance)

    def _check_odom_timeout(self):
        if self._last_odom_time is None:
            return
        if rospy.get_time() - self._last_odom_time > self.tracking_loss_timeout:
            self._set_tracking(True, 'timeout')

    def _update_tf_correction(self):
        try:
            tr = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0),
                                                 rospy.Duration(0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return
        t = tr.transform.translation
        q = tr.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        if self._prev_tf is not None:
            self.tf_dx = t.x - self._prev_tf[0]
            self.tf_dy = t.y - self._prev_tf[1]
            self.tf_dyaw = yaw - self._prev_tf[2]
        self._prev_tf = (t.x, t.y, yaw)

    # ---------------- main loop ----------------
    def run(self):
        while not rospy.is_shutdown():
            self._update_tf_correction()
            self._check_odom_timeout()

            t = rospy.get_time()
            row = [
                t,
                self.pitch, self.roll, self.yaw,
                self.accel[0], self.accel[1], self.accel[2],
                self.ang_vel[0], self.ang_vel[1], self.ang_vel[2],
                self.ang_acc_x, self.ang_acc_z,
                self.odom[0], self.odom[1], self.odom[2],
                self.odom_rpy[0], self.odom_rpy[1], self.odom_rpy[2],
                self.odom_vx, self.odom_vyaw,
                self.cmd_vx, self.cmd_vyaw,
                self.gt[0], self.gt[1], self.gt[2],
                self.gt_rpy[0], self.gt_rpy[1], self.gt_rpy[2],
                self.gt_v[0], self.gt_v[1], self.gt_distance,
                self.loop_closures, self.proximity,
                self.inliers, self.matches,
                self.icp_ratio, self.icp_corr, self.proc_time,
                self.tracking_lost, self.tracking_loss_count,
                self.relocalization_count,
                math.hypot(self.tf_dx, self.tf_dy), self.tf_dyaw,
            ]
            for link in self.wheel_links:
                row.append(self.contact[link])
                row.append(self.normal_force[link])

            self.rows.append(row)

            if self.gt_ready:
                self.gt_poses.append((t,) + tuple(self.gt) + tuple(self.gt_quat))

            self.rate.sleep()

    # ---------------- output ----------------
    def save(self):
        if not self.rows:
            rospy.logwarn('No data collected; nothing written.')
            return

        path = os.path.join(self.run_dir, 'metrics.csv')
        with open(path, 'w') as f:
            w = csvmod.writer(f)
            w.writerow(csv_header(self.columns))
            w.writerows(self.rows)

        write_units_reference(self.columns,
                              os.path.join(self.run_dir, 'columns.csv'))
        write_tum(os.path.join(self.run_dir, 'gt_traj.tum'), self.gt_poses)
        write_tum(os.path.join(self.run_dir, 'est_traj.tum'), self.est_poses)

        with open(os.path.join(self.run_dir, 'events.csv'), 'w') as f:
            w = csvmod.writer(f)
            w.writerow(['timestamp[s]', 'event[-]', 'detail[-]',
                        'gt_x[m]', 'gt_y[m]', 'gt_distance[m]'])
            w.writerows(self.events)

        self._write_meta()

        rospy.loginfo('Run %s/%02d saved to %s',
                      self.robot_name, self.run_id, self.run_dir)
        rospy.loginfo('  %d samples, %d GT poses, %d events',
                      len(self.rows), len(self.gt_poses), len(self.events))
        rospy.loginfo('  distance travelled (GT): %.2f m', self.gt_distance)
        rospy.loginfo('  tracking losses: %d, relocalizations: %d',
                      self.tracking_loss_count, self.relocalization_count)
        rospy.loginfo('  loop closures: %d', self.loop_closures)

    def _write_meta(self):
        duration = (self.rows[-1][0] - self.rows[0][0]) if self.rows else 0.0
        meta = {
            'robot': self.robot_name,
            'run_id': self.run_id,
            'gazebo_model': self.model_name,
            'samples': len(self.rows),
            'duration_s': round(duration, 3),
            'gt_distance_m': round(self.gt_distance, 3),
            'log_rate_hz': 50,
            'ground_truth': {
                'topic': '/gazebo/model_states',
                'type': 'gazebo_msgs/ModelStates',
                'filtering': 'none',
                'added_noise': 'none',
                'frame': 'gazebo world frame, model canonical link',
                'timestamping': 'ROS sim clock at reception (no header stamp '
                                'on ModelStates); error bounded by one 1 ms '
                                'physics step',
            },
            'slam': {
                'loop_closures': self.loop_closures,
                'proximity_detections': self.proximity,
                'tracking_losses': self.tracking_loss_count,
                'relocalizations': self.relocalization_count,
            },
            'wheel_links': self.wheel_links,
            'ros_time_is_sim_time': rospy.get_param('/use_sim_time', False),
            'git_commit': self._git_commit(),
        }
        with open(os.path.join(self.run_dir, 'run_meta.yaml'), 'w') as f:
            yaml.safe_dump(meta, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _git_commit():
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            out = subprocess.check_output(
                ['git', '-C', here, 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return 'unknown'


if __name__ == '__main__':
    try:
        MetricsLogger().run()
    except rospy.ROSInterruptException:
        pass
