#!/usr/bin/env python3
"""cmd_vel -> four wheel velocities (Husky, skid-steer).

Counterpart of twist_to_tracks.py on the tracked platform and of
twist_to_wheels.py on the Rocker-Bogie.  It exists because the Husky no longer
runs libgazebo_ros_skid_steer_drive.so: that plugin drove the ODE joint motors
directly, which made it a rigid velocity source with zero droop under load and
therefore not comparable with the PID velocity loops of the other two
platforms.  The Husky now runs the same gazebo_ros_control loop they do, and
something has to turn a Twist into the four joint velocity commands it takes.

KINEMATICS
==========
The Husky is a 4-wheel skid-steer: both wheels on a side turn together and a
yaw rate is produced by the speed difference between the two sides, the lateral
slip being absorbed by the contact patches.  So there are only two independent
commands,

    v_left  = (v - w * B / 2) / r
    v_right = (v + w * B / 2) / r

with B the track width and r the wheel radius, and each is published to both
wheels on that side.  B is the wheelSeparation the old plugin used (0.5709 m),
and r is the wheel collision radius from the model (0.17775 m); both are
recorded in sim_config.yaml and reported by extract_robot_specs.py.

WHAT THIS NODE DELIBERATELY DOES NOT DO
=======================================
No minimum angular velocity.  The tracked platform's node forces |w| up to
0.25 rad/s whenever a turn is commanded, to overcome track friction; that is a
per-platform advantage on exactly the turn-in-place behaviour the RPE metric
measures, and sim_config.yaml now pins min_angular to 0.0 for all three.  The
parameter is exposed so the asymmetry can be reproduced deliberately, and the
node warns loudly if it is ever set non-zero.

No acceleration limiting.  The old plugin applied a 2.0 m/s^2 command slew rate
that neither of the other platforms had.  Acceleration is bounded by the shared
torque budget and by the shared DWA acc_lim_x, and by nothing else.
"""

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class TwistToWheels(object):
    def __init__(self):
        rospy.init_node('twist_to_wheels', anonymous=False)

        self.track_width = rospy.get_param('~track_width', 0.5709)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.17775)
        # Kept at 0.0 by sim_config.yaml; see the module docstring.
        self.min_angular = rospy.get_param('~min_angular', 0.0)

        self.pubs_left = [
            rospy.Publisher('wheel_vel_controller_fl/command',
                            Float64, queue_size=10),
            rospy.Publisher('wheel_vel_controller_rl/command',
                            Float64, queue_size=10),
        ]
        self.pubs_right = [
            rospy.Publisher('wheel_vel_controller_fr/command',
                            Float64, queue_size=10),
            rospy.Publisher('wheel_vel_controller_rr/command',
                            Float64, queue_size=10),
        ]

        self.sub = rospy.Subscriber('cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo('twist_to_wheels (Husky, skid-steer) started')
        rospy.loginfo('  track_width  = %.4f m', self.track_width)
        rospy.loginfo('  wheel_radius = %.5f m', self.wheel_radius)
        if self.min_angular > 0.0:
            rospy.logwarn('  min_angular = %.2f rad/s is ACTIVE: this platform '
                          'will turn harder than commanded, which breaks the '
                          'command-path parity declared in sim_config.yaml',
                          self.min_angular)

    def cmd_vel_callback(self, msg):
        v = msg.linear.x
        w = msg.angular.z

        if self.min_angular > 0.0 and 0.01 < abs(w) < self.min_angular:
            w = self.min_angular if w > 0 else -self.min_angular

        half = self.track_width / 2.0
        v_left = (v - w * half) / self.wheel_radius
        v_right = (v + w * half) / self.wheel_radius

        for pub in self.pubs_left:
            pub.publish(Float64(v_left))
        for pub in self.pubs_right:
            pub.publish(Float64(v_right))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        TwistToWheels().run()
    except rospy.ROSInterruptException:
        pass
