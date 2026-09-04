#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

from romea_localisation_msgs.msg import ObservationPosition2DStamped
from romea_mobile_base_msgs.msg import OneAxleSteeringMeasureStamped

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException
)

from scipy.spatial.transform import Rotation


class KalmanPomOdometryGPS(Node):

    def __init__(self):
        super().__init__('kalman_pom_odometry_gps')

        # State [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

        self.initialized = False
        self.last_odom_time = None

        # Filter parameters
        self.q = 0.5

        self.odom_covariance_scale = 1.0
        self.gps_covariance_scale = 1.0

        self.default_odom_variance = 0.01
        self.default_gps_variance_x = 0.01
        self.default_gps_variance_y = 0.01

        # TF2 temporal compensation
        self.odom_tf_delay = 0.10
        self.gps_tf_delay = 0.11

        self.tf_timeout = 0.05

        # GPS observation matrix
        self.H_gps = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # TF2 library
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True
        )

        # QoS
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Subscribers
        self.odom_sub = self.create_subscription(
            OneAxleSteeringMeasureStamped,
            '/robot/base/controller/odometry',
            self.odom_callback,
            qos
        )

        self.gps_sub = self.create_subscription(
            ObservationPosition2DStamped,
            '/robot/localisation/position',
            self.gps_callback,
            qos
        )

        # Publishers
        self.position_publisher = self.create_publisher(
            PoseStamped,
            '/robot/localisation/estimated_position_stamped',
            10
        )

        self.covariance_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot/localisation/estimated_position_covariance',
            10
        )

        self.state_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot/localisation/estimated_state',
            10
        )

        self.get_logger().info(
            'KF Sabi: odometry prediction + GPS correction'
        )

        self.get_logger().info(
            f'Odom TF delay = {self.odom_tf_delay:.3f} s'
        )

        self.get_logger().info(
            f'GPS TF delay = {self.gps_tf_delay:.3f} s'
        )

    # TF2 orientation
    def get_tf_yaw(self, stamp):

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'robot_base_link',
                stamp,
                timeout=Duration(
                    seconds=self.tf_timeout
                )
            )

        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException
        ) as e:

            self.get_logger().warning(
                f'TF2 lookup failed: {e}'
            )

            return None, None

        q = transform.transform.rotation

        rotation = Rotation.from_quat([
            q.x,
            q.y,
            q.z,
            q.w
        ])

        theta = rotation.as_euler(
            'xyz'
        )[2]

        return theta, rotation

    # Odometry prediction
    def odom_callback(self, msg):

        if not self.initialized:
            return

        current_time = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        if self.last_odom_time is None:
            self.last_odom_time = current_time
            return

        dt = (
            current_time
            - self.last_odom_time
        )

        if dt <= 0.0 or dt > 0.5:

            self.last_odom_time = current_time
            return

        # TF timestamp compensation
        stamp = (
            Time.from_msg(msg.header.stamp)
            - Duration(
                seconds=self.odom_tf_delay
            )
        )

        theta, _ = self.get_tf_yaw(
            stamp
        )

        if theta is None:
            return

        self.last_odom_time = current_time

        # Odometry longitudinal velocity
        speed = float(
            msg.measure.longitudinal_speed
        )

        # Robot velocity -> map frame
        vx = (
            speed
            * math.cos(theta)
        )

        vy = (
            speed
            * math.sin(theta)
        )

        # Odometry-driven prediction
        self.x[0, 0] += vx * dt
        self.x[1, 0] += vy * dt

        self.x[2, 0] = vx
        self.x[3, 0] = vy

        # State-transition Jacobian
        F = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        # Odometry covariance
        cov = msg.measure.covariance

        if (
            len(cov) > 0
            and np.isfinite(cov[0])
            and cov[0] > 0.0
        ):
            var_v = float(
                cov[0]
            )

        else:
            var_v = (
                self.default_odom_variance
            )

        var_v *= (
            self.odom_covariance_scale
        )

        c = math.cos(theta)
        s = math.sin(theta)

        # Velocity covariance in map frame
        R_v = var_v * np.array([
            [c * c, c * s],
            [c * s, s * s]
        ])

        # Noise injection matrix
        G = np.array([
            [dt, 0.0],
            [0.0, dt],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        Q_odom = (
            G
            @ R_v
            @ G.T
        )

        # Additional model uncertainty
        Q_model = np.diag([
            self.q * dt,
            self.q * dt,
            self.q * dt,
            self.q * dt
        ])

        # Covariance prediction
        self.P = (
            F
            @ self.P
            @ F.T
            + Q_odom
            + Q_model
        )

        self.P = 0.5 * (
            self.P
            + self.P.T
        )

        self.publish_state(
            msg.header.stamp,
            theta
        )

    # GPS correction
    def gps_callback(self, msg):

        x_gps = float(
            msg.observation_position.position.x
        )

        y_gps = float(
            msg.observation_position.position.y
        )

        # GPS lever arm
        lx = float(
            msg.observation_position.level_arm.x
        )

        ly = float(
            msg.observation_position.level_arm.y
        )

        lz = float(
            msg.observation_position.level_arm.z
        )

        # TF timestamp compensation
        stamp = (
            Time.from_msg(msg.header.stamp)
            - Duration(
                seconds=self.gps_tf_delay
            )
        )

        theta, rotation = self.get_tf_yaw(
            stamp
        )

        if theta is None:

            self.get_logger().warning(
                'TF2 transformation unavailable, '
                'GPS skipped'
            )

            return

        # Lever-arm compensation
        lever_arm_map = rotation.apply(
            np.array([
                lx,
                ly,
                lz
            ])
        )

        x_meas = (
            x_gps
            - lever_arm_map[0]
        )

        y_meas = (
            y_gps
            - lever_arm_map[1]
        )

        # Initialization
        if not self.initialized:

            self.x[0, 0] = x_meas
            self.x[1, 0] = y_meas

            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0

            self.last_odom_time = (
                msg.header.stamp.sec
                + msg.header.stamp.nanosec
                * 1e-9
            )

            self.initialized = True

            self.get_logger().info(
                'KF initialized from GPS'
            )

            self.publish_state(
                msg.header.stamp,
                theta
            )

            return

        # GPS covariance
        cov = (
            msg.observation_position
            .position.covariance
        )

        cov_x = float(
            cov[0]
        )

        cov_xy = float(
            cov[1]
        )

        cov_yx = float(
            cov[2]
        )

        cov_y = float(
            cov[3]
        )

        if (
            not np.isfinite(cov_x)
            or cov_x <= 0.0
        ):

            cov_x = (
                self.default_gps_variance_x
            )

            cov_xy = 0.0

        if (
            not np.isfinite(cov_y)
            or cov_y <= 0.0
        ):

            cov_y = (
                self.default_gps_variance_y
            )

            cov_yx = 0.0

        R = np.array([
            [cov_x, cov_xy],
            [cov_yx, cov_y]
        ])

        R *= (
            self.gps_covariance_scale
        )

        R += (
            np.eye(2)
            * 1e-9
        )

        Z = np.array([
            [x_meas],
            [y_meas]
        ])

        # Innovation
        innovation = (
            Z
            - self.H_gps
            @ self.x
        )

        # Innovation covariance
        S = (
            self.H_gps
            @ self.P
            @ self.H_gps.T
            + R
        )

        # Kalman gain
        try:

            K = np.linalg.solve(
                S.T,
                (
                    self.P
                    @ self.H_gps.T
                ).T
            ).T

        except np.linalg.LinAlgError:

            K = (
                self.P
                @ self.H_gps.T
                @ np.linalg.pinv(S)
            )

        # GPS correction
        self.x = (
            self.x
            + K
            @ innovation
        )

        # Joseph covariance update
        I = np.eye(4)

        IKH = (
            I
            - K
            @ self.H_gps
        )

        self.P = (
            IKH
            @ self.P
            @ IKH.T
            + K
            @ R
            @ K.T
        )

        self.P = 0.5 * (
            self.P
            + self.P.T
        )

        self.publish_state(
            msg.header.stamp,
            theta
        )

    # Publications
    def publish_state(
        self,
        stamp,
        theta
    ):

        if not self.initialized:
            return

        pose = PoseStamped()

        pose.header.stamp = stamp
        pose.header.frame_id = 'map'

        pose.pose.position.x = float(
            self.x[0, 0]
        )

        pose.pose.position.y = float(
            self.x[1, 0]
        )

        pose.pose.position.z = 0.0

        pose.pose.orientation.z = (
            math.sin(
                theta / 2.0
            )
        )

        pose.pose.orientation.w = (
            math.cos(
                theta / 2.0
            )
        )

        self.position_publisher.publish(
            pose
        )

        covariance = Float64MultiArray()

        covariance.data = [
            float(self.P[0, 0]),
            float(self.P[0, 1]),
            float(self.P[1, 0]),
            float(self.P[1, 1])
        ]

        self.covariance_publisher.publish(
            covariance
        )

        state = Float64MultiArray()

        state.data = [
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0])
        ]

        self.state_publisher.publish(
            state
        )


def main(args=None):

    rclpy.init(args=args)

    node = KalmanPomOdometryGPS()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
