#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class TwistToTracks:
    def __init__(self):
        rospy.init_node('twist_to_tracks', anonymous=False)

        self.track_separation = rospy.get_param('~track_separation', 0.5709)
        self.sprocket_radius = rospy.get_param('~sprocket_radius', 0.178)
        self.min_angular = rospy.get_param('~min_angular', 0.25)

        self.pub_left = rospy.Publisher(
            'sprocket_velocity_controller_left/command', Float64, queue_size=10)
        self.pub_right = rospy.Publisher(
            'sprocket_velocity_controller_right/command', Float64, queue_size=10)

        self.sub = rospy.Subscriber('cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo("twist_to_tracks iniciado")
        rospy.loginfo(f"  track_separation: {self.track_separation}m")
        rospy.loginfo(f"  sprocket_radius: {self.sprocket_radius}m")

    def cmd_vel_callback(self, msg):
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Reforzar velocidad angular minima para vencer friccion de orugas
        if abs(angular_z) > 0.01 and abs(angular_z) < self.min_angular:
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
