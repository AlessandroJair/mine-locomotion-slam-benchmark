#!/usr/bin/env python3

import rospy
import math
import tf
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion


class WheelOdom:
    def __init__(self):
        rospy.init_node('wheel_odom', anonymous=False)

        self.wheel_separation = rospy.get_param('~wheel_separation', 0.6775)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.178)

        # Joints: left = rev_14(fl), rev_13(ml), rev_12(rl)
        #         right = rev_9(fr), rev_10(mr), rev_11(rr)
        self.left_joints = ['rev_14', 'rev_13', 'rev_12']
        self.right_joints = ['rev_9', 'rev_10', 'rev_11']

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = None

        # IMU roll/pitch (updated by IMU callback)
        self.imu_roll = 0.0
        self.imu_pitch = 0.0

        # Publishers
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=50)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Subscribers
        rospy.Subscriber('joint_states', JointState, self.joint_states_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)

        rospy.loginfo("wheel_odom iniciado (sep=%.3f, rad=%.3f) con IMU roll/pitch",
                      self.wheel_separation, self.wheel_radius)

    def imu_cb(self, msg):
        # Extract roll and pitch from IMU orientation
        q = msg.orientation
        euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.imu_roll = euler[0]
        self.imu_pitch = euler[1]

    def joint_states_cb(self, msg):
        now = msg.header.stamp
        if now.to_sec() == 0:
            now = rospy.Time.now()

        # Build name-to-velocity map
        vel_map = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.velocity):
                vel_map[name] = msg.velocity[i]

        # Average left and right wheel angular velocities
        left_vels = [vel_map[j] for j in self.left_joints if j in vel_map]
        right_vels = [vel_map[j] for j in self.right_joints if j in vel_map]

        if not left_vels or not right_vels:
            return

        avg_left = sum(left_vels) / len(left_vels)
        avg_right = sum(right_vels) / len(right_vels)

        # Convert to linear velocities (negated for inverted robot)
        v_left = -avg_left * self.wheel_radius
        v_right = -avg_right * self.wheel_radius

        # Differential drive kinematics
        linear_vel = (v_right + v_left) / 2.0
        angular_vel = (v_right - v_left) / self.wheel_separation

        # Integrate position (2D)
        if self.last_time is not None:
            dt = (now - self.last_time).to_sec()
            if dt > 0 and dt < 1.0:
                delta_theta = angular_vel * dt
                delta_x = linear_vel * math.cos(self.theta + delta_theta / 2.0) * dt
                delta_y = linear_vel * math.sin(self.theta + delta_theta / 2.0) * dt

                self.x += delta_x
                self.y += delta_y
                self.theta += delta_theta

        self.last_time = now

        # Quaternion with roll/pitch from IMU + yaw from wheels
        odom_quat = tf.transformations.quaternion_from_euler(
            self.imu_roll, self.imu_pitch, self.theta)

        # Publish TF: odom -> base_link (includes real roll/pitch)
        self.tf_broadcaster.sendTransform(
            (self.x, self.y, 0.0),
            odom_quat,
            now,
            "rocker_bogie/base_link_nav",
            "odom"
        )

        # Publish Odometry message
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "rocker_bogie/base_link_nav"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = Quaternion(*odom_quat)

        odom.twist.twist.linear.x = linear_vel
        odom.twist.twist.angular.z = angular_vel

        self.odom_pub.publish(odom)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = WheelOdom()
        node.run()
    except rospy.ROSInterruptException:
        pass
