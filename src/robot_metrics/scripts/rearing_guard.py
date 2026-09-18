#!/usr/bin/env python3
"""Cancel the current move_base goal when the vehicle rears against something.

WHY THIS EXISTS
===============
Both differential rollovers of 2026-08-20 had the same shape.  The vehicle
drove into a face it could not climb, the wheels kept turning because a
velocity was still commanded, and the traction torque pitched the chassis up
until it passed its tip-over angle:

    step descent    stalled 10.8 s, body rose 0.661 m, pitch reached -43.7 deg
    south-west      stalled 28.7 s, body rose 0.237 m, pitch reached -44.5 deg

Nothing intervened.  move_base's own escape is oscillation_timeout, which only
fires once the robot has failed to cover oscillation_distance (0.5 m) for the
whole timeout; at 60 s that was useless, and even at 15 s the step-descent
rearing was over in 9 s.  This node reacts in about a second.

WHAT IT WATCHES
===============
Rearing is a stall plus a rising pitch, and it needs both: a genuine climb also
shows a large pitch, but the vehicle is moving while it does it.

    stalled  = a speed is commanded and the vehicle is not making it
    reared   = |pitch| past a threshold well below the tip-over angle

Speed comes from the SLAM odometry, not from the wheels.  Wheel odometry is
exactly wrong here: the wheels are spinning, so it reports the motion that is
not happening.

WHAT IT DOES
============
Publishes an empty GoalID on move_base/cancel, which aborts the current goal.
The waypoint navigator then moves on, the same as it does for any other
failure.  It does not fight the controller for cmd_vel and it does not latch:
once the pose recovers it re-arms.

Thresholds are parameters so the same node serves all three platforms; keeping
it one shared node is deliberate, because a guard that behaved differently per
platform would be another un-equalised test condition.
"""

import math

import rospy
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def rpy(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    return math.atan2(sinr, cosr), math.asin(sinp)


class RearingGuard(object):

    def __init__(self):
        cmd_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        odom_topic = rospy.get_param('~odom_topic', '/rtabmap/odom')
        imu_topic = rospy.get_param('~imu_topic', '/imu/data')

        # A speed is being asked for ...
        self.cmd_min = rospy.get_param('~commanded_min', 0.15)      # m/s
        # ... and this little of it is arriving.
        self.speed_max = rospy.get_param('~speed_max', 0.05)        # m/s
        # Hold both for this long before believing it.
        self.stall_time = rospy.get_param('~stall_time', 1.0)       # s
        # Pitch that counts as reared, once stalled.  Well under the ~37-40 deg
        # at which these chassis actually go over.
        self.pitch_warn = rospy.get_param('~pitch_warn', 15.0)      # deg
        # Pitch that is an emergency whatever else is true.
        self.pitch_abort = rospy.get_param('~pitch_abort', 30.0)    # deg
        # Do not spam cancellations.
        self.cooldown = rospy.get_param('~cooldown', 5.0)           # s

        self.cmd_vx = 0.0
        self.speed = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.stalled_since = None
        self.last_cancel = -1e9
        self.cancels = 0

        # Cancelling alone is not enough.  Measured 2026-08-21: the
        # differential wedged on the 0.10 m step at (53.41, -27.73), nose up
        # 48.1 deg, roll 0.0, and moved 9 mm in 25 s.  The guard was firing the
        # whole time - 48 deg is well past pitch_abort - but a cancelled goal
        # only stops the pushing; it does not put the vehicle back down.  The
        # navigator then sent the next waypoint and it pushed again.
        # So the guard also backs the vehicle off what it is climbing.  It only
        # does this after cancelling, so move_base has stopped publishing and
        # the two are not fighting for the topic.
        self.backoff_speed = rospy.get_param('~backoff_speed', -0.25)   # m/s
        self.backoff_time = rospy.get_param('~backoff_time', 2.0)       # s
        self.backing_until = 0.0

        self.pub = rospy.Publisher('move_base/cancel', GoalID, queue_size=1)
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(cmd_topic, Twist, self.cmd_cb)
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        rospy.Subscriber(imu_topic, Imu, self.imu_cb)

        rospy.loginfo('rearing_guard: cmd=%s odom=%s imu=%s', cmd_topic,
                      odom_topic, imu_topic)
        rospy.loginfo('rearing_guard: stall if |cmd_vx|>%.2f and speed<%.2f for '
                      '%.1fs; rear at %.0f deg; abort outright at %.0f deg',
                      self.cmd_min, self.speed_max, self.stall_time,
                      self.pitch_warn, self.pitch_abort)
        rospy.on_shutdown(self.report)

    # ---------------- inputs ----------------
    def cmd_cb(self, msg):
        self.cmd_vx = msg.linear.x

    def odom_cb(self, msg):
        v = msg.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)

    def imu_cb(self, msg):
        self.roll, self.pitch = (math.degrees(a)
                                 for a in rpy(msg.orientation))

    # ---------------- decision ----------------
    def step(self):
        now = rospy.get_time()

        # Backing off: drive it, and let nothing else run until it is done.
        if now < self.backing_until:
            tw = Twist()
            tw.linear.x = self.backoff_speed
            self.cmd_pub.publish(tw)
            return
        if 0.0 < self.backing_until <= now:
            self.cmd_pub.publish(Twist())      # one explicit stop
            self.backing_until = 0.0
            self.stalled_since = None
            return

        stalled = abs(self.cmd_vx) > self.cmd_min and self.speed < self.speed_max
        if stalled:
            if self.stalled_since is None:
                self.stalled_since = now
        else:
            self.stalled_since = None

        held = (self.stalled_since is not None
                and now - self.stalled_since >= self.stall_time)
        reason = None
        if abs(self.pitch) >= self.pitch_abort:
            reason = ('pitch %.1f deg past the %.0f deg abort threshold'
                      % (self.pitch, self.pitch_abort))
        elif held and abs(self.pitch) >= self.pitch_warn:
            reason = ('reared: %.1f s stalled (cmd %.2f m/s, moving %.2f m/s) '
                      'at pitch %.1f deg'
                      % (now - self.stalled_since, self.cmd_vx, self.speed,
                         self.pitch))

        if reason and now - self.last_cancel >= self.cooldown:
            self.last_cancel = now
            self.cancels += 1
            self.pub.publish(GoalID())
            self.backing_until = now + self.backoff_time
            rospy.logwarn('rearing_guard: cancelling goal and backing off '
                          '%.2f m/s for %.1f s - %s',
                          self.backoff_speed, self.backoff_time, reason)
            self.stalled_since = None

    def report(self):
        rospy.loginfo('rearing_guard: intervened %d time(s)', self.cancels)

    def run(self):
        rate = rospy.Rate(rospy.get_param('~rate', 20.0))
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('rearing_guard')
    try:
        RearingGuard().run()
    except rospy.ROSInterruptException:
        pass
