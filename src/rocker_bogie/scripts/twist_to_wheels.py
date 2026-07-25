#!/usr/bin/env python3
"""cmd_vel -> six wheel velocities + four steering angles (Rocker-Bogie).

The Rocker-Bogie in this study is NOT a skid-steer platform.  It has four
steered corner wheels (front and rear, counter-steering) and two unsteered
middle wheels, so a turn is produced by steering geometry, not by forcing a
velocity difference between the two sides.  This node implements that.

FRAMES
======
cmd_vel arrives in the standard REP-103 body frame: +x forward, +y left, +z up,
yaw counter-clockwise.  The URDF's base_link is authored facing -x, and the
robot is spawned at Y = pi to compensate.  A 180 deg rotation about z maps

    x -> -x        y -> -y        z -> z

so longitudinal and lateral quantities change sign between the two frames but
ANGLES ABOUT Z DO NOT.  Two consequences, and getting them mixed up is what
made the previous version steer and drive inconsistently:

  * steering angles need NO sign correction - a steering joint's axis is z in
    both frames, so the nav-frame Ackermann angle is published as-is;
  * wheel spin does: a positive spin about the joint's +y axis rolls the robot
    toward base_link +x, which is BACKWARD.  So the joint velocity is the
    negative of the nav-frame wheel speed over the radius.  That is the single
    frame flip in this file.

Everything below is computed in the nav frame, and wheels are addressed by
their physical nav-frame position (fl, fr, ml, mr, rl, rr).  The launch file
binds those names to the correct joints; see the mapping table there.

GEOMETRY
========
Wheel centres in base_link at zero configuration, measured from the URDF by
extract_robot_specs.py.  Recall FRONT = smaller x, LEFT = smaller y:

    front-left  Rueda_6_1 (-0.143, -0.182)   front-right Rueda_1_1 (-0.143, +0.414)
    middle-left Rueda_4_1 (+0.283, -0.275)   middle-right Rueda_3_1 (+0.283, +0.507)
    rear-left   Rueda_2_1 (+0.710, -0.275)   rear-right  Rueda_5_1 (+0.710, +0.507)

    wheelbase   = 0.8532 m   front axle to rear axle
    front_track = 0.5959 m   the steered front pair is inset
    rear_track  = 0.7821 m   middle and rear pairs share this track
    wheel_radius= 0.178  m

The front and rear axles are symmetric about x = +0.2835, which is exactly the
middle axle.  Counter-steering the two steered axles by equal and opposite
angles therefore places the instantaneous centre of rotation on the middle-axle
line - the same line the LiDAR sits on - which is what makes the turn clean for
SLAM.

Note that the front and rear tracks differ, so the Ackermann angles are NOT the
same on the two axles for a given radius; each axle is solved with its own
track.  A single averaged track (what the previous version used) biases the
inner and outer angles on both axles at once.

ACKERMANN SOLUTION
==================
For a turn of signed radius R measured from the ICR to the vehicle centreline,
with L = wheelbase and t the track of the axle being solved, the steer angle of
the wheel on the inner/outer side is

    delta = atan( (L/2) / (R -/+ t/2) )

applied with opposite sign on the rear axle.  The wheel speeds are scaled by
the radius each wheel actually traces about the ICR,

    v_i = v * r_i / R,      r_corner = hypot(L/2, R -/+ t/2)
                            r_middle = |R -/+ t/2|

so that no wheel is dragged.  When |R| exceeds max_radius the solution
degenerates to a straight line and all steering angles go to zero.

SPIN TURNS
==========
When linear.x is (near) zero and angular.z is not, an Ackermann solution does
not exist: the requested ICR is inside the wheelbase.  The node then steers all
four corner wheels to the tangent of the circle centred on the robot and drives
the two sides in opposite directions.  This is the only mode in which the
platform uses a velocity difference, and it is reported as such.

CONVENTIONS
===========
cmd_vel is interpreted in the standard REP-103 body frame: +x forward, +z yaw
counter-clockwise.  base_link is authored facing -x, so the sign of the
longitudinal command is flipped once, here, and nowhere else.
"""

import math

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class TwistToWheels(object):

    def __init__(self):
        rospy.init_node('twist_to_wheels', anonymous=False)

        # --- geometry (metres) ---
        self.wheelbase = rospy.get_param('~wheelbase', 0.8532)
        self.front_track = rospy.get_param('~front_track', 0.5961)
        self.rear_track = rospy.get_param('~rear_track', 0.7821)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.178)

        # --- limits ---
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 0.5)  # rad
        # Beyond this radius the turn is treated as straight (m).
        self.max_radius = rospy.get_param('~max_radius', 200.0)
        # Deadbands on the incoming command.
        self.lin_deadband = rospy.get_param('~lin_deadband', 0.01)   # m/s
        self.ang_deadband = rospy.get_param('~ang_deadband', 0.01)   # rad/s

        # Floor on the spin-turn rate.  Historically 0.45 rad/s, to break
        # static friction when the suspension was locked and the steering
        # geometry was wrong.  It silently made the robot rotate faster than
        # commanded, which corrupts both the yaw-rate and the slip metrics, so
        # it now defaults to OFF.  Raise it only if a spin turn actually
        # stalls, and report the value if you do.
        self.min_angular = rospy.get_param('~min_angular', 0.0)      # rad/s

        self.half_wb = self.wheelbase / 2.0
        self.half_ft = self.front_track / 2.0
        self.half_rt = self.rear_track / 2.0

        # Wheel positions in the NAV frame: (x forward, y left).
        self.geom = {
            'fl': (+self.half_wb, +self.half_ft),
            'fr': (+self.half_wb, -self.half_ft),
            'ml': (0.0, +self.half_rt),
            'mr': (0.0, -self.half_rt),
            'rl': (-self.half_wb, +self.half_rt),
            'rr': (-self.half_wb, -self.half_rt),
        }
        self.steered = ('fl', 'fr', 'rl', 'rr')

        # --- publishers: 6 wheel velocities ---
        self.pub_v = {
            k: rospy.Publisher('wheel_vel_controller_{}/command'.format(k),
                               Float64, queue_size=1)
            for k in ('fl', 'fr', 'ml', 'mr', 'rl', 'rr')
        }
        # --- publishers: 4 steering angles ---
        self.pub_s = {
            k: rospy.Publisher('steer_controller_{}/command'.format(k),
                               Float64, queue_size=1)
            for k in ('fl', 'fr', 'rl', 'rr')
        }

        rospy.Subscriber('cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo('twist_to_wheels (Ackermann, 4-wheel counter-steer)')
        rospy.loginfo('  wheelbase   = %.4f m', self.wheelbase)
        rospy.loginfo('  front track = %.4f m', self.front_track)
        rospy.loginfo('  rear track  = %.4f m', self.rear_track)
        rospy.loginfo('  wheel radius= %.4f m', self.wheel_radius)
        rospy.loginfo('  max steer   = %.3f rad (%.1f deg)',
                      self.max_steer_angle, math.degrees(self.max_steer_angle))
        rospy.loginfo('  min turn radius at full lock = %.3f m',
                      self.half_wb / math.tan(self.max_steer_angle))
        if self.min_angular > 0.0:
            rospy.logwarn('  min_angular = %.2f rad/s is ACTIVE: spin turns '
                          'will execute faster than commanded.',
                          self.min_angular)

    # ------------------------------------------------------------------
    def clamp_steer(self, angle):
        return max(-self.max_steer_angle, min(self.max_steer_angle, angle))

    def publish(self, vel, steer):
        for k, v in vel.items():
            self.pub_v[k].publish(Float64(v))
        for k, s in steer.items():
            self.pub_s[k].publish(Float64(s))

    def stop(self):
        self.publish(dict.fromkeys(self.pub_v, 0.0),
                     dict.fromkeys(self.pub_s, 0.0))

    def to_joint_velocity(self, ground_speed):
        """Nav-frame wheel ground speed [m/s] -> joint velocity [rad/s].

        This is the one place where the 180 deg model heading is applied.
        """
        return -ground_speed / self.wheel_radius

    def go_straight(self, v_nav):
        w = self.to_joint_velocity(v_nav)
        self.publish(dict.fromkeys(self.pub_v, w),
                     dict.fromkeys(self.pub_s, 0.0))

    # ------------------------------------------------------------------
    def cmd_vel_callback(self, msg):
        v = msg.linear.x        # nav frame, +x forward
        w = msg.angular.z       # nav frame, +z CCW

        moving = abs(v) > self.lin_deadband
        turning = abs(w) > self.ang_deadband

        if not moving and not turning:
            self.stop()
            return

        if not moving and turning:
            self.spin_turn(w)
            return

        if not turning:
            self.go_straight(v)
            return

        # Signed turn radius from the ICR to the vehicle centre.  R > 0 puts
        # the ICR on the left, so the left wheels are the inner ones.
        radius = v / w
        if abs(radius) > self.max_radius:
            self.go_straight(v)
            return

        self.ackermann(v, radius)

    # ------------------------------------------------------------------
    def ackermann(self, v_nav, radius):
        """Signed-radius Ackermann for two counter-steering axles.

        Working with a signed radius removes the special-casing by turn
        direction that the previous version needed - and with it the chance of
        assigning inner and outer to the wrong sides.  For a wheel whose nav
        lateral offset is y (left positive), the ICR is (R - y) away from that
        wheel's longitudinal line, and the inner wheel is simply the one with
        the smallest |R - y|.
        """
        R = radius
        steer = {}
        vel = {}

        for key, (x, y) in self.geom.items():
            d = R - y
            # Radius this wheel traces about the ICR.
            r_wheel = math.hypot(x, d) if key in self.steered else abs(d)
            # Ground speed, scaled by the radius it traces.
            ground_speed = v_nav * r_wheel / abs(R)
            vel[key] = self.to_joint_velocity(ground_speed)

            if key in self.steered:
                # delta = atan(x / d): the sign is carried by both x (front
                # positive, rear negative, giving the counter-steer for free)
                # and d (turn direction).
                steer[key] = self.clamp_steer(math.atan(x / d))

        self.publish(vel, steer)

    # ------------------------------------------------------------------
    def spin_turn(self, w):
        """Rotate about the robot centre; no Ackermann solution exists.

        The requested ICR lies inside the wheelbase, so no steering angle can
        make every wheel roll cleanly.  The corner wheels are turned as close
        to tangent as the steering lock allows and the two sides are driven in
        opposite directions - the one manoeuvre in which this platform does
        rely on a velocity difference.
        """
        if self.min_angular > 0.0 and abs(w) < self.min_angular:
            w = math.copysign(self.min_angular, w)

        steer = {}
        vel = {}
        for key, (x, y) in self.geom.items():
            # Pure rotation about the centre: v = w * z_hat x p, so
            # v_x = -w*y and v_y = +w*x.
            vx = -w * y
            vy = w * x

            # Tangent direction, clamped to the steering lock.  Middle wheels
            # cannot steer, so they stay at zero.
            delta = self.clamp_steer(math.atan2(vy, vx)) if key in self.steered else 0.0
            if key in self.steered:
                steer[key] = delta

            # Roll at the projection of the demanded velocity onto the
            # direction the wheel is actually pointing.  Projecting rather
            # than using |v| matters because the lock clamps delta: the
            # unreachable component is scrubbed off sideways either way, but
            # projecting at least keeps the rolling component consistent
            # instead of over-driving every wheel.
            vel[key] = self.to_joint_velocity(
                vx * math.cos(delta) + vy * math.sin(delta))

        self.publish(vel, steer)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        TwistToWheels().run()
    except rospy.ROSInterruptException:
        pass
