#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from romea_localisation_msgs.msg import ObservationPosition2DStamped

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException
)


class KalmanDog(Node):

    def __init__(self):
        super().__init__('kalman_dog')

        # Frames
        self.map_frame = 'map'
        self.base_frame = 'robot_base_footprint'

        # State [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

        # Filter parameters
        self.q = 1e-4
        self.maximum_dt = 0.5

        self.odom_covariance_scale = 1.0
        self.gps_covariance_scale = 4.7

        self.default_odom_variance_vx = 0.01
        self.default_odom_variance_vy = 0.01

        self.default_gps_variance_x = 0.0036
        self.default_gps_variance_y = 0.0036

        # TF timing
        self.odom_tf_delay = 0.10
        self.gps_tf_delay = 0.11
        self.tf_timeout = 0.05

        self.initialized = False
        self.last_odom_time = None

        # GPS observation matrix
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # TF2
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True
        )

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/robot/base/controller/odom',
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
            '/dog_kalman/estimated_position_stamped',
            10
        )

        self.covariance_publisher = self.create_publisher(
            Float64MultiArray,
            '/dog_kalman/estimated_position_covariance',
            10
        )

        self.state_publisher = self.create_publisher(
            Float64MultiArray,
            '/dog_kalman/estimated_state',
            10
        )

        self.get_logger().info(
            'Dog KF started: odometry prediction + GPS correction'
        )

        self.get_logger().info(
            'State: [x, y, vx, vy]'
        )

        self.get_logger().info(
            'TF2 yaw: map -> robot_base_footprint'
        )

    # Get yaw from TF2
    def get_yaw_from_tf(self, stamp, delay):

        tf_stamp = (
            Time.from_msg(stamp)
            - Duration(seconds=delay)
        )

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                tf_stamp,
                timeout=Duration(
                    seconds=self.tf_timeout
                )
            )

        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException
        ):
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

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

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

        if dt <= 0.0 or dt > self.maximum_dt:
            self.last_odom_time = current_time
            return

        theta = self.get_yaw_from_tf(
            msg.header.stamp,
            self.odom_tf_delay
        )

        if theta is None:
            return

        self.last_odom_time = current_time

        # Odometry velocity in robot frame
        vx_robot = float(
            msg.twist.twist.linear.x
        )

        vy_robot = float(
            msg.twist.twist.linear.y
        )

        # Robot frame -> map frame
        c = math.cos(theta)
        s = math.sin(theta)

        vx_map = (
            vx_robot * c
            - vy_robot * s
        )

        vy_map = (
            vx_robot * s
            + vy_robot * c
        )

        # Prediction
        self.x[0, 0] += vx_map * dt
        self.x[1, 0] += vy_map * dt

        self.x[2, 0] = vx_map
        self.x[3, 0] = vy_map

        # State transition matrix
        A = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Process noise
        Q = self.q * np.array([
            [
                dt**4 / 4.0,
                0.0,
                dt**3 / 2.0,
                0.0
            ],
            [
                0.0,
                dt**4 / 4.0,
                0.0,
                dt**3 / 2.0
            ],
            [
                dt**3 / 2.0,
                0.0,
                dt**2,
                0.0
            ],
            [
                0.0,
                dt**3 / 2.0,
                0.0,
                dt**2
            ]
        ])

        # Odometry covariance
        cov = msg.twist.covariance

        var_vx = float(cov[0])
        var_vy = float(cov[7])

        if (
            not np.isfinite(var_vx)
            or var_vx <= 0.0
        ):
            var_vx = self.default_odom_variance_vx

        if (
            not np.isfinite(var_vy)
            or var_vy <= 0.0
        ):
            var_vy = self.default_odom_variance_vy

        R_body = np.array([
            [var_vx, 0.0],
            [0.0, var_vy]
        ])

        # Rotate odometry covariance into map frame
        rotation = np.array([
            [c, -s],
            [s,  c]
        ])

        R_velocity = (
            rotation
            @ R_body
            @ rotation.T
        )

        R_velocity *= (
            self.odom_covariance_scale
        )

        # Inject velocity uncertainty
        G = np.array([
            [dt, 0.0],
            [0.0, dt],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        Q_odom = (
            G
            @ R_velocity
            @ G.T
        )

        # Covariance prediction
        self.P = (
            A
            @ self.P
            @ A.T
            + Q
            + Q_odom
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

        theta = self.get_yaw_from_tf(
            msg.header.stamp,
            self.gps_tf_delay
        )

        if theta is None:
            return

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

        # Rotate lever arm into map frame
        c = math.cos(theta)
        s = math.sin(theta)

        lx_map = (
            lx * c
            - ly * s
        )

        ly_map = (
            lx * s
            + ly * c
        )

        # GPS antenna -> robot base
        x_meas = (
            x_gps
            - lx_map
        )

        y_meas = (
            y_gps
            - ly_map
        )

        current_time = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        # Initialization
        if not self.initialized:

            self.x[0, 0] = x_meas
            self.x[1, 0] = y_meas

            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0

            self.last_odom_time = current_time

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

        var_x = float(cov[0])
        cov_xy = float(cov[1])
        cov_yx = float(cov[2])
        var_y = float(cov[3])

        if (
            not np.isfinite(var_x)
            or var_x <= 0.0
        ):
            var_x = self.default_gps_variance_x
            cov_xy = 0.0

        if (
            not np.isfinite(var_y)
            or var_y <= 0.0
        ):
            var_y = self.default_gps_variance_y
            cov_yx = 0.0

        R = np.array([
            [var_x, cov_xy],
            [cov_yx, var_y]
        ])

        R *= self.gps_covariance_scale

        R += np.eye(2) * 1e-9

        # Measurement
        z = np.array([
            [x_meas],
            [y_meas]
        ])

        # Innovation
        innovation = (
            z
            - self.H @ self.x
        )

        # Innovation covariance
        S = (
            self.H
            @ self.P
            @ self.H.T
            + R
        )

        # Kalman gain
        try:
            K = np.linalg.solve(
                S.T,
                (
                    self.P
                    @ self.H.T
                ).T
            ).T

        except np.linalg.LinAlgError:
            K = (
                self.P
                @ self.H.T
                @ np.linalg.pinv(S)
            )

        # State correction
        self.x = (
            self.x
            + K @ innovation
        )

        # Joseph covariance update
        I = np.eye(4)

        IKH = (
            I
            - K @ self.H
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

    # Publish estimate
    def publish_state(self, stamp, theta):

        if not self.initialized:
            return

        pose = PoseStamped()

        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame

        pose.pose.position.x = float(
            self.x[0, 0]
        )

        pose.pose.position.y = float(
            self.x[1, 0]
        )

        pose.pose.position.z = 0.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0

        pose.pose.orientation.z = (
            math.sin(theta / 2.0)
        )

        pose.pose.orientation.w = (
            math.cos(theta / 2.0)
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

    node = KalmanDog()

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
