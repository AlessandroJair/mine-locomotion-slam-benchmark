#!/usr/bin/env python3
"""El arco de alineacion lo dispara el rumbo respecto a la RUTA, no al objetivo.

    source /opt/ros/noetic/setup.bash   # trajectory_follower importa rospy
    python3 test_align_trigger.py
"""
import math
import os
import sys

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
sys.path.insert(0, SCRIPTS)
from trajectory_follower import fuera_de_rumbo  # noqa

# recta hacia +y, como la bajada de s = 95-110 m
PTS = [(18.3, -26.0 + 0.25 * k) for k in range(40)]
TH = 0.8
NORTE = math.pi / 2


def main():
    # 1.3 m al lado y a 6 deg de la ruta: NO es un rumbo malo (era el fallo)
    assert not fuera_de_rumbo(NORTE + math.radians(6), PTS, 10, TH)
    # atravesado a 90 deg: si
    assert fuera_de_rumbo(0.0, PTS, 10, TH)
    # justo por encima y por debajo del umbral, en los dos sentidos
    assert fuera_de_rumbo(NORTE + TH + 0.01, PTS, 10, TH)
    assert not fuera_de_rumbo(NORTE - TH + 0.01, PTS, 10, TH)
    # rumbo contrario, cruzando +-pi
    assert fuera_de_rumbo(-NORTE, PTS, 10, TH)
    # ultimo punto y tramo degenerado no revientan ni disparan
    assert not fuera_de_rumbo(NORTE, PTS, len(PTS) - 1, TH)
    assert not fuera_de_rumbo(0.0, [(0.0, 0.0), (0.0, 0.0)], 0, TH)
    print('test_align_trigger: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
