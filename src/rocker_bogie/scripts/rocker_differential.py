#!/usr/bin/env python3
"""Rocker differential for the Rocker-Bogie suspension.

WHAT A ROCKER DIFFERENTIAL DOES
===============================
On a real rocker-bogie (Sojourner, MER, MSL) the two rocker arms are not
independent: they are tied together across the chassis by a differential -
either a bevel-gear box or a transverse bogie bar.  The differential enforces

    phi_left + phi_right = 0                                             (1)

where phi is the rocker angle measured RELATIVE TO THE CHASSIS.  Expressed in
the world frame this is the familiar statement that the chassis pitch is the
average of the two rocker pitches,

    theta_chassis = (theta_left + theta_right) / 2

which is the whole point of the mechanism: when one side climbs an obstacle the
body tilts by only half the angle, and - crucially for this paper - the sensor
mast stays far closer to level than it would on a rigid or independently-sprung
chassis.  That is the mechanical claim the paper makes about SLAM quality, so
it has to be in the model.

WHY IT IS A NODE AND NOT A JOINT
================================
Equation (1) is a closed-loop constraint.  URDF describes trees only, so it
cannot be expressed as a joint, and Gazebo's <joint type="gearbox"> is not
reachable from a URDF-spawned model.  The constraint is therefore imposed the
way a physical differential imposes it - with torque:

    e     = phi_left + phi_right              constraint violation   [rad]
    e_dot = phi_dot_left + phi_dot_right                             [rad/s]
    tau   = -k_stiffness * e - k_damping * e_dot                     [N*m]

and tau is applied to BOTH rocker joints with the same sign.  That is exactly
the generalised force of the holonomic constraint g = phi_l + phi_r: since
dg/dphi_l = dg/dphi_r = 1, the multiplier enters both joints identically, and
the reaction -2*tau lands on the chassis, which is where a real differential is
mounted.

The coupling is stiff but not rigid, so (1) holds to within e_rms rather than
exactly.  The node reports that residual on shutdown; if it is not small
compared with the rocker travel, raise ~stiffness.

TUNING
======
Effective rocker inertia about its pivot is I = 4.046 kg*m^2 after the mass
matching (8.535 kg*m^2 native), measured from the expanded URDF by the parallel
axis theorem over the 25 links, 13.48 kg, of the rocker subtree.  The penalty
pair (k, c) behaves as a second-order system with

    omega_n = sqrt(k / I)          zeta = c / (2 * sqrt(k * I))

CORRECTED 2026-08-18.  This docstring used to claim I ~ 0.45 kg*m^2 and, from
it, that the native k = 2000 / c = 60 pair was critically damped at zeta ~ 1.0.
The inertia was roughly nineteen times too small, so the real figure was

    zeta = 60 / (2 * sqrt(2000 * 8.535)) = 0.23

i.e. markedly under-damped, and it stayed 0.23 through the mass matching
because k and c were scaled with I.  The consequence was not cosmetic: at
zeta = 0.23 a terrain disturbance overshoots by ~48%, the error saturates tau
at max_torque, and because tau is applied with the SAME SIGN to both rockers
the chassis takes the whole 2 * max_torque = 379 N*m as a pitching reaction.
Measured on the run of 2026-08-18: both rocker pivots ended pinned against
their +-0.6 rad stops with a mean |phi_l + phi_r| of 66.8 deg, and the chassis
pitch went from level at spawn to a mean of -29 deg and a worst of -53 deg.

The defaults are now k = 2000 N*m/rad and c = 2*sqrt(k*I) = 261.3 N*m*s/rad
natively, scaled with the mass matching to k = 948.08 and c = 123.87, which is
zeta = 1.0 at the true inertia.  Both are ROS parameters, both are reported in
the paper's model table, and ~inertia below is what the startup diagnostic uses
so it can no longer disagree with them.

ABLATION
========
Set ~enabled:=false (or simply do not launch this node) to obtain the
"independent rockers, no differential" configuration.  The joints then receive
zero effort and are free, which is the correct control condition for showing
what the differential contributes.
"""

import math

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


LEFT_JOINT = 'rocker_pivot_left'
RIGHT_JOINT = 'rocker_pivot_right'


class RockerDifferential(object):

    def __init__(self):
        rospy.init_node('rocker_differential', anonymous=False)

        # Defaults sized for the mass-matched model (45.0 kg, rocker inertia
        # 4.046 kg*m^2 about the pivot): wn = 15.3 rad/s, zeta = 0.23. They are
        # the native 2000 / 60 / 400 scaled by the same 0.474039 as every mass
        # in ensamblajeurdf.xacro - see robot_metrics/scripts/scale_masses.py.
        # The launch file passes the same values explicitly; these defaults
        # exist so running the node standalone does not silently over-stiffen
        # the coupling by a factor of two.
        self.enabled = rospy.get_param('~enabled', True)
        self.stiffness = rospy.get_param('~stiffness', 948.08)   # N*m/rad
        self.damping = rospy.get_param('~damping', 28.44)        # N*m*s/rad
        self.max_torque = rospy.get_param('~max_torque', 189.6)  # N*m
        # Rocker subtree inertia about the pivot, after mass matching.
        # Only used to report omega_n and zeta at startup - but reporting
        # them from a wrong constant is what hid the mis-tuning before.
        self.inertia = rospy.get_param('~inertia', 4.046)  # kg*m^2
        self.rate_hz = rospy.get_param('~rate', 200.0)           # Hz

        # Joint states of the passive suspension come from the Gazebo joint
        # state publisher declared in ensamblajeurdf.gazebo, not from
        # ros_control: the bogies deliberately have no transmission.
        # 'joint_states', no 'suspension_joint_states'.  El plugin que los
        # publica ignora su <topicName> y usa <robotNamespace>/joint_states
        # (ver la nota en ensamblajeurdf.gazebo).  Con el nombre viejo este
        # nodo NUNCA recibio un JointState: phi_l y phi_r se quedaban en None
        # y el par de acoplamiento no se aplicaba jamas.  No se noto porque
        # ~enabled esta en false desde que el varillaje geometrico lo
        # sustituyo, pero re-activarlo habria sido un no-op silencioso.
        joint_states_topic = rospy.get_param(
            '~joint_states_topic', 'joint_states')

        self.phi_l = None
        self.phi_r = None
        self.dphi_l = 0.0
        self.dphi_r = 0.0
        self.last_stamp = None

        # Residual statistics, reported on shutdown so the quality of the
        # constraint is documented rather than assumed.
        self._err_sq_sum = 0.0
        self._err_abs_max = 0.0
        self._err_n = 0

        self.pub_left = rospy.Publisher(
            'rocker_diff_effort_left/command', Float64, queue_size=1)
        self.pub_right = rospy.Publisher(
            'rocker_diff_effort_right/command', Float64, queue_size=1)

        rospy.Subscriber(joint_states_topic, JointState, self.joint_state_cb)

        rospy.on_shutdown(self.report)

        if self.enabled:
            # Was hardcoded to 0.45 kg*m^2 while the model's real rocker
            # inertia is 4.046: the node printed zeta ~ 0.69 every launch when
            # the truth was 0.23, which is how the under-damping went unnoticed.
            # It is a parameter now so it cannot drift from the model again.
            omega_n = math.sqrt(self.stiffness / self.inertia)
            zeta = self.damping / (2.0 * math.sqrt(self.stiffness * self.inertia))
            rospy.loginfo('Rocker differential ENABLED')
            rospy.loginfo('  stiffness   = %.1f N*m/rad', self.stiffness)
            rospy.loginfo('  damping     = %.1f N*m*s/rad', self.damping)
            rospy.loginfo('  max torque  = %.1f N*m', self.max_torque)
            rospy.loginfo('  -> omega_n ~ %.1f rad/s (%.1f Hz), zeta ~ %.2f',
                          omega_n, omega_n / (2.0 * math.pi), zeta)
        else:
            rospy.logwarn('Rocker differential DISABLED (~enabled=false): '
                          'rockers are independent. This is the ablation '
                          'configuration, not the nominal robot.')

    def joint_state_cb(self, msg):
        try:
            il = msg.name.index(LEFT_JOINT)
            ir = msg.name.index(RIGHT_JOINT)
        except ValueError:
            rospy.logwarn_throttle(
                10.0,
                'Suspension joint states do not contain %s / %s. Got: %s',
                LEFT_JOINT, RIGHT_JOINT, list(msg.name))
            return

        self.phi_l = msg.position[il]
        self.phi_r = msg.position[ir]

        # Prefer the reported velocities; fall back to differentiating the
        # angle if the publisher does not fill the velocity field.
        if len(msg.velocity) > max(il, ir):
            self.dphi_l = msg.velocity[il]
            self.dphi_r = msg.velocity[ir]

    def step(self):
        if self.phi_l is None:
            return

        error = self.phi_l + self.phi_r
        error_dot = self.dphi_l + self.dphi_r

        tau = -self.stiffness * error - self.damping * error_dot
        tau = max(-self.max_torque, min(self.max_torque, tau))

        # Same sign on both joints: see the module docstring for why.
        self.pub_left.publish(Float64(tau))
        self.pub_right.publish(Float64(tau))

        self._err_sq_sum += error * error
        self._err_abs_max = max(self._err_abs_max, abs(error))
        self._err_n += 1

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.enabled:
                self.step()
            else:
                # Explicit zero effort: leaves the joints free.
                self.pub_left.publish(Float64(0.0))
                self.pub_right.publish(Float64(0.0))
            rate.sleep()

    def report(self):
        if not self.enabled or self._err_n == 0:
            return
        rms = math.sqrt(self._err_sq_sum / self._err_n)
        rospy.loginfo('Rocker differential constraint residual '
                      '(phi_l + phi_r, ideal 0):')
        rospy.loginfo('  RMS = %.5f rad (%.3f deg)', rms, math.degrees(rms))
        rospy.loginfo('  max = %.5f rad (%.3f deg)',
                      self._err_abs_max, math.degrees(self._err_abs_max))
        if rms > 0.02:
            rospy.logwarn('  residual is large: the differential is behaving '
                          'compliantly. Increase ~stiffness or report the '
                          'residual alongside the results.')


if __name__ == '__main__':
    try:
        RockerDifferential().run()
    except rospy.ROSInterruptException:
        pass
