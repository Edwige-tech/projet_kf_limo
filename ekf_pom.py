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


class EkfPom(Node):

    def __init__(self):
        super().__init__('ekf_pom')

        # Robot parameters
        self.wheelbase = 2.0
        self.maximum_dt = 0.5

        # TF2 used only for initial yaw
        self.gps_tf_delay = 0.11
        self.tf_timeout = 0.05

        # Covariance parameters
        self.declare_parameter(
            'use_gps_for_initial_position_covariance',
            True
        )

        self.declare_parameter('p0_x', 1.0)
        self.declare_parameter('p0_y', 1.0)
        self.declare_parameter('p0_yaw', 0.10)

        self.declare_parameter(
            'use_odometry_covariance',
            True
        )

        self.declare_parameter(
            'sigma_speed',
            0.10
        )

        self.declare_parameter(
            'sigma_steering',
            0.03
        )

        self.declare_parameter(
            'q_position_x',
            1e-4
        )

        self.declare_parameter(
            'q_position_y',
            1e-4
        )

        self.declare_parameter(
            'q_yaw',
            1e-4
        )

        self.declare_parameter(
            'use_gps_covariance',
            True
        )

        self.declare_parameter(
            'gps_variance_x',
            0.01
        )

        self.declare_parameter(
            'gps_variance_y',
            0.01
        )

        self.declare_parameter(
            'gps_covariance_xy',
            0.0
        )

        # State X = [x, y, theta]
        self.x = np.zeros((3, 1))

        self.P = np.diag([
            1.0,
            1.0,
            0.5
        ])

        self.initialized = False
        self.last_odometry_stamp = None

        # TF2 only for initialization
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True
        )

        # QoS
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Odometry prediction
        self.odometry_subscriber = self.create_subscription(
            OneAxleSteeringMeasureStamped,
            '/robot/base/controller/odometry',
            self.odometry_callback,
            qos
        )

        # GPS correction
        self.position_subscriber = self.create_subscription(
            ObservationPosition2DStamped,
            '/robot/localisation/position',
            self.position_callback,
            qos
        )

        # Publishers
        self.position_publisher = self.create_publisher(
            PoseStamped,
            '/robot/localisation/ekf_position_stamped',
            10
        )

        self.covariance_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot/localisation/ekf_position_covariance',
            10
        )

        self.get_logger().info(
            'EKF POM: odometry prediction + GPS correction'
        )

    # Angle normalization
    def normalize_angle(self, angle):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    # Initial yaw from TF2
    def get_initial_yaw(self, stamp):

        tf_stamp = (
            stamp
            - Duration(
                seconds=self.gps_tf_delay
            )
        )

        try:

            transform = self.tf_buffer.lookup_transform(
                'map',
                'robot_base_link',
                tf_stamp,
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
                f'TF2 initialization failed: {e}'
            )

            return None

        q = transform.transform.rotation

        siny_cosp = 2.0 * (
            q.w * q.z
            + q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y
            + q.z * q.z
        )

        theta = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        return theta

    # Odometry callback
    def odometry_callback(self, msg):

        if not self.initialized:
            return

        current_stamp = Time.from_msg(
            msg.header.stamp
        )

        if self.last_odometry_stamp is None:

            self.last_odometry_stamp = (
                current_stamp
            )

            return

        dt = (
            current_stamp
            - self.last_odometry_stamp
        ).nanoseconds / 1e9

        self.last_odometry_stamp = (
            current_stamp
        )

        if (
            dt <= 0.0
            or dt > self.maximum_dt
        ):
            return

        speed = float(
            msg.measure.longitudinal_speed
        )

        steering = float(
            msg.measure.steering_angle
        )

        self.predict(
            speed,
            steering,
            dt,
            msg.measure.covariance
        )

        self.publish_state(
            msg.header.stamp
        )

    # EKF prediction
    def predict(
        self,
        speed,
        steering,
        dt,
        covariance
    ):

        theta = float(
            self.x[2, 0]
        )

        steering = np.clip(
            steering,
            -math.radians(70.0),
            math.radians(70.0)
        )

        # Odometry projected using estimated theta
        vx = (
            speed
            * math.cos(theta)
        )

        vy = (
            speed
            * math.sin(theta)
        )

        yaw_rate = (
            speed
            / self.wheelbase
            * math.tan(steering)
        )

        # Nonlinear prediction
        self.x[0, 0] += (
            vx
            * dt
        )

        self.x[1, 0] += (
            vy
            * dt
        )

        self.x[2, 0] = (
            self.normalize_angle(
                theta
                + yaw_rate
                * dt
            )
        )

        # State Jacobian
        F = np.array([
            [
                1.0,
                0.0,
                -speed
                * math.sin(theta)
                * dt
            ],
            [
                0.0,
                1.0,
                speed
                * math.cos(theta)
                * dt
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ])

        cos_steering = max(
            math.cos(steering) ** 2,
            1e-8
        )

        # Input Jacobian
        G = np.array([
            [
                math.cos(theta) * dt,
                0.0
            ],
            [
                math.sin(theta) * dt,
                0.0
            ],
            [
                math.tan(steering)
                / self.wheelbase
                * dt,

                speed
                / (
                    self.wheelbase
                    * cos_steering
                )
                * dt
            ]
        ])

        Q_input = (
            self.get_odometry_covariance(
                covariance
            )
        )

        qx = float(
            self.get_parameter(
                'q_position_x'
            ).value
        )

        qy = float(
            self.get_parameter(
                'q_position_y'
            ).value
        )

        qtheta = float(
            self.get_parameter(
                'q_yaw'
            ).value
        )

        Q_model = np.diag([
            qx * dt,
            qy * dt,
            qtheta * dt
        ])

        Q = (
            G
            @ Q_input
            @ G.T
            + Q_model
        )

        # Covariance prediction
        self.P = (
            F
            @ self.P
            @ F.T
            + Q
        )

        self.P = 0.5 * (
            self.P
            + self.P.T
        )

    # Odometry covariance
    def get_odometry_covariance(
        self,
        covariance
    ):

        if self.get_parameter(
            'use_odometry_covariance'
        ).value:

            if len(covariance) >= 4:

                Q = np.array([
                    [
                        float(covariance[0]),
                        float(covariance[1])
                    ],
                    [
                        float(covariance[2]),
                        float(covariance[3])
                    ]
                ])

                Q = 0.5 * (
                    Q
                    + Q.T
                )

                if (
                    np.all(np.isfinite(Q))
                    and Q[0, 0] > 0.0
                    and Q[1, 1] > 0.0
                ):
                    return Q

        sigma_v = float(
            self.get_parameter(
                'sigma_speed'
            ).value
        )

        sigma_delta = float(
            self.get_parameter(
                'sigma_steering'
            ).value
        )

        return np.diag([
            sigma_v ** 2,
            sigma_delta ** 2
        ])

    # GPS callback
    def position_callback(self, msg):

        x_gps = float(
            msg.observation_position.position.x
        )

        y_gps = float(
            msg.observation_position.position.y
        )

        lx = float(
            msg.observation_position.level_arm.x
        )

        ly = float(
            msg.observation_position.level_arm.y
        )

        R = (
            self.get_gps_covariance(msg)
            + np.eye(2) * 1e-9
        )

        # Initialization
        if not self.initialized:

            stamp = Time.from_msg(
                msg.header.stamp
            )

            theta0 = self.get_initial_yaw(
                stamp
            )

            if theta0 is None:

                self.get_logger().warning(
                    'EKF initialization postponed'
                )

                return

            c = math.cos(theta0)
            s = math.sin(theta0)

            # Base position from GPS antenna position
            x0 = (
                x_gps
                - lx * c
                + ly * s
            )

            y0 = (
                y_gps
                - lx * s
                - ly * c
            )

            self.x[:, 0] = [
                x0,
                y0,
                self.normalize_angle(theta0)
            ]

            if self.get_parameter(
                'use_gps_for_initial_position_covariance'
            ).value:

                self.P = np.diag([
                    max(
                        R[0, 0],
                        1e-4
                    ),
                    max(
                        R[1, 1],
                        1e-4
                    ),
                    float(
                        self.get_parameter(
                            'p0_yaw'
                        ).value
                    )
                ])

            else:

                self.P = np.diag([
                    float(
                        self.get_parameter(
                            'p0_x'
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            'p0_y'
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            'p0_yaw'
                        ).value
                    )
                ])

            self.initialized = True
            self.last_odometry_stamp = None

            self.get_logger().info(
                'EKF initialized from GPS and initial TF2 yaw'
            )

            self.publish_state(
                msg.header.stamp
            )

            return

        # GPS correction using estimated theta
        self.update_gps(
            x_gps,
            y_gps,
            lx,
            ly,
            R
        )

        self.publish_state(
            msg.header.stamp
        )

    # GPS covariance
    def get_gps_covariance(
        self,
        msg
    ):

        if self.get_parameter(
            'use_gps_covariance'
        ).value:

            covariance = (
                msg.observation_position
                .position.covariance
            )

            if len(covariance) >= 4:

                R = np.array([
                    [
                        float(covariance[0]),
                        float(covariance[1])
                    ],
                    [
                        float(covariance[2]),
                        float(covariance[3])
                    ]
                ])

                R = 0.5 * (
                    R
                    + R.T
                )

                if (
                    np.all(np.isfinite(R))
                    and R[0, 0] > 0.0
                    and R[1, 1] > 0.0
                ):
                    return R

        rx = float(
            self.get_parameter(
                'gps_variance_x'
            ).value
        )

        ry = float(
            self.get_parameter(
                'gps_variance_y'
            ).value
        )

        rxy = float(
            self.get_parameter(
                'gps_covariance_xy'
            ).value
        )

        return np.array([
            [rx, rxy],
            [rxy, ry]
        ])

    # Nonlinear GPS correction
    def update_gps(
        self,
        x_gps,
        y_gps,
        lx,
        ly,
        R
    ):

        x_est = float(
            self.x[0, 0]
        )

        y_est = float(
            self.x[1, 0]
        )

        theta = float(
            self.x[2, 0]
        )

        c = math.cos(theta)
        s = math.sin(theta)

        # Predicted GPS antenna position
        h = np.array([
            [
                x_est
                + lx * c
                - ly * s
            ],
            [
                y_est
                + lx * s
                + ly * c
            ]
        ])

        # Raw GPS antenna measurement
        z = np.array([
            [x_gps],
            [y_gps]
        ])

        # GPS observation Jacobian
        H = np.array([
            [
                1.0,
                0.0,
                -lx * s
                - ly * c
            ],
            [
                0.0,
                1.0,
                lx * c
                - ly * s
            ]
        ])

        # Innovation
        innovation = (
            z
            - h
        )

        # Innovation covariance
        S = (
            H
            @ self.P
            @ H.T
            + R
        )

        # Kalman gain
        try:

            K = np.linalg.solve(
                S.T,
                (
                    self.P
                    @ H.T
                ).T
            ).T

        except np.linalg.LinAlgError:

            K = (
                self.P
                @ H.T
                @ np.linalg.pinv(S)
            )

        # State correction
        self.x = (
            self.x
            + K
            @ innovation
        )

        self.x[2, 0] = (
            self.normalize_angle(
                float(
                    self.x[2, 0]
                )
            )
        )

        # Joseph covariance update
        I = np.eye(3)

        IKH = (
            I
            - K
            @ H
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

    # Publications
    def publish_state(
        self,
        stamp
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

        theta = float(
            self.x[2, 0]
        )

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


def main(args=None):

    rclpy.init(args=args)

    node = EkfPom()

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
