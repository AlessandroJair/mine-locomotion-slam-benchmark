#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class TwistToTracks:
    def __init__(self):
        rospy.init_node('twist_to_tracks', anonymous=False)

        self.track_separation = rospy.get_param('~track_separation', 0.5709)
        self.sprocket_radius = rospy.get_param('~sprocket_radius', 0.148)
        # 0.0 for all three platforms; see command.min_angular_rad_s in
        # robot_metrics/config/sim_config.yaml.  This used to default to
        # 0.25 rad/s to overcome track friction, which meant this platform
        # turned harder than it was asked to whenever a small yaw rate was
        # commanded - a per-platform advantage on exactly the turn-in-place
        # behaviour the RPE metric is sensitive to, and one the Husky and
        # the Rocker-Bogie never had.  The joint drag it was compensating
        # for is now zero and integral action covers the rest.
        self.min_angular = rospy.get_param('~min_angular', 0.0)

        self.pub_left = rospy.Publisher(
            'sprocket_velocity_controller_left/command', Float64, queue_size=10)
        self.pub_right = rospy.Publisher(
            'sprocket_velocity_controller_right/command', Float64, queue_size=10)

        self.sub = rospy.Subscriber('cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo("twist_to_tracks iniciado")
        rospy.loginfo(f"  track_separation: {self.track_separation}m")
        rospy.loginfo(f"  sprocket_radius: {self.sprocket_radius}m")
        if self.min_angular > 0.0:
            rospy.logwarn("  min_angular = %.2f rad/s is ACTIVE: this "
                          "platform will turn harder than commanded, "
                          "which breaks the command-path parity declared "
                          "in sim_config.yaml", self.min_angular)

    def cmd_vel_callback(self, msg):
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Off by default - see min_angular above.
        if self.min_angular > 0.0 and 0.01 < abs(angular_z) < self.min_angular:
            angular_z = self.min_angular if angular_z > 0 else -self.min_angular

        v_left = (linear_x - angular_z * self.track_separation / 2.0) / self.sprocket_radius
        v_right = (linear_x + angular_z * self.track_separation / 2.0) / self.sprocket_radius

        self.pub_left.publish(Float64(v_left))
        self.pub_right.publish(Float64(v_right))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = TwistToTracks()
        node.run()
    except rospy.ROSInterruptException:
        pass
