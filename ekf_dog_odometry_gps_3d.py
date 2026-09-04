#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from romea_localisation_msgs.msg import ObservationPosition2DStamped


class EkfDogOdometryGPS3D(Node):

    def __init__(self):
        super().__init__('ekf_dog_odometry_gps_3d')

        self.map_frame = 'map'

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
            10
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

        self.state_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot/localisation/ekf_state',
            10
        )

        # Parameters
        self.lever_arm_x = 0.1

        self.q_position = 1e-4
        self.q_yaw = 1e-4

        self.odom_covariance_scale = 1.0

        self.default_odom_variance_vx = 0.01
        self.default_odom_variance_vy = 0.01
        self.default_odom_variance_omega = 0.01

        self.default_gps_variance_x = 0.0036
        self.default_gps_variance_y = 0.0036

        self.declare_parameter('gps_covariance_scale', 4.7)
        self.gps_covariance_scale = float(
            self.get_parameter('gps_covariance_scale').value
        )

        # State X = [x, y, theta]^T
        self.x = np.zeros((3, 1))
        self.P = np.eye(3)

        self.initialized = False
        self.first_gps_x = None
        self.first_gps_y = None
        self.last_time = None

        # GPS measures x and y
        self.H_gps = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        self.get_logger().info("3D EKF started")
        self.get_logger().info("State: [x, y, theta]")
        self.get_logger().info(
            f"GPS covariance scale = {self.gps_covariance_scale}"
        )

    # Angle normalization
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    # EKF prediction
    def odom_callback(self, msg):

        if not self.initialized or self.last_time is None:
            return

        current_time = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        dt = current_time - self.last_time

        if dt <= 0.0 or dt > 0.5:
            self.last_time = current_time
            return

        self.last_time = current_time

        # Odometry inputs
        v_robot_x = float(msg.twist.twist.linear.x)
        v_robot_y = float(msg.twist.twist.linear.y)
        omega = float(msg.twist.twist.angular.z)

        theta = float(self.x[2, 0])

        c = math.cos(theta)
        s = math.sin(theta)

        # Robot velocities -> map frame
        v_map_x = v_robot_x * c - v_robot_y * s
        v_map_y = v_robot_x * s + v_robot_y * c

        # State prediction
        self.x[0, 0] += v_map_x * dt
        self.x[1, 0] += v_map_y * dt
        self.x[2, 0] = self.normalize_angle(theta + omega * dt)

        # Jacobian F = df/dx
        F = np.array([
            [
                1.0,
                0.0,
                (-v_robot_x * s - v_robot_y * c) * dt
            ],
            [
                0.0,
                1.0,
                (v_robot_x * c - v_robot_y * s) * dt
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ])

        # Noise injection matrix G = df/du
        G = np.array([
            [c * dt, -s * dt, 0.0],
            [s * dt,  c * dt, 0.0],
            [0.0,     0.0,    dt]
        ])

        # Odometry covariance
        cov = msg.twist.covariance

        cov_vx = float(cov[0])
        cov_vy = float(cov[7])
        cov_omega = float(cov[35])

        if not np.isfinite(cov_vx) or cov_vx <= 0.0:
            cov_vx = self.default_odom_variance_vx

        if not np.isfinite(cov_vy) or cov_vy <= 0.0:
            cov_vy = self.default_odom_variance_vy

        if not np.isfinite(cov_omega) or cov_omega <= 0.0:
            cov_omega = self.default_odom_variance_omega

        Q_input = self.odom_covariance_scale * np.diag([
            cov_vx,
            cov_vy,
            cov_omega
        ])

        Q_model = np.diag([
            self.q_position * dt,
            self.q_position * dt,
            self.q_yaw * dt
        ])

        Q = G @ Q_input @ G.T + Q_model

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

        self.publish_state(msg.header.stamp)

    # GPS correction
    def gps_callback(self, msg):

        x_raw = float(msg.observation_position.position.x)
        y_raw = float(msg.observation_position.position.y)

        current_time = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        # First GPS point
        if self.first_gps_x is None:
            self.first_gps_x = x_raw
            self.first_gps_y = y_raw
            self.last_time = current_time
            return

        # Initial heading from first GPS displacement
        if not self.initialized:

            dx = x_raw - self.first_gps_x
            dy = y_raw - self.first_gps_y

            if math.hypot(dx, dy) < 0.05:
                return

            theta = math.atan2(dy, dx)

            x_gps = x_raw - self.lever_arm_x * math.cos(theta)
            y_gps = y_raw - self.lever_arm_x * math.sin(theta)

            self.x[:, 0] = [
                x_gps,
                y_gps,
                theta
            ]

            self.last_time = current_time
            self.initialized = True

            self.get_logger().info(
                f"EKF initialized, heading = "
                f"{math.degrees(theta):.2f} deg"
            )

            return

        theta = float(self.x[2, 0])

        # Lever arm compensation
        x_gps = x_raw - self.lever_arm_x * math.cos(theta)
        y_gps = y_raw - self.lever_arm_x * math.sin(theta)

        Z = np.array([
            [x_gps],
            [y_gps]
        ])

        # GPS covariance
        cov = msg.observation_position.position.covariance

        cov_x = float(cov[0])
        cov_xy = float(cov[1])
        cov_yx = float(cov[2])
        cov_y = float(cov[3])

        if not np.isfinite(cov_x) or cov_x <= 0.0:
            cov_x = self.default_gps_variance_x
            cov_xy = 0.0

        if not np.isfinite(cov_y) or cov_y <= 0.0:
            cov_y = self.default_gps_variance_y
            cov_yx = 0.0

        R = np.array([
            [cov_x, cov_xy],
            [cov_yx, cov_y]
        ])

        R *= self.gps_covariance_scale
        R += np.eye(2) * 1e-9

        # Innovation
        innovation = Z - self.H_gps @ self.x
        S = self.H_gps @ self.P @ self.H_gps.T + R

        # Kalman gain
        try:
            K = np.linalg.solve(
                S.T,
                (self.P @ self.H_gps.T).T
            ).T
        except np.linalg.LinAlgError:
            K = self.P @ self.H_gps.T @ np.linalg.pinv(S)

        # State correction
        self.x = self.x + K @ innovation
        self.x[2, 0] = self.normalize_angle(
            float(self.x[2, 0])
        )

        # Joseph covariance update
        I = np.eye(3)
        IKH = I - K @ self.H_gps

        self.P = (
            IKH @ self.P @ IKH.T
            + K @ R @ K.T
        )

        self.P = 0.5 * (self.P + self.P.T)

    # Publications
    def publish_state(self, stamp):

        if not self.initialized:
            return

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame

        pose.pose.position.x = float(self.x[0, 0])
        pose.pose.position.y = float(self.x[1, 0])
        pose.pose.position.z = 0.0

        theta = float(self.x[2, 0])

        pose.pose.orientation.z = math.sin(theta / 2.0)
        pose.pose.orientation.w = math.cos(theta / 2.0)

        self.position_publisher.publish(pose)

        covariance = Float64MultiArray()
        covariance.data = [
            float(self.P[0, 0]),
            float(self.P[0, 1]),
            float(self.P[1, 0]),
            float(self.P[1, 1])
        ]

        self.covariance_publisher.publish(covariance)

        state = Float64MultiArray()
        state.data = [
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0])
        ]

        self.state_publisher.publish(state)


def main(args=None):

    rclpy.init(args=args)

    node = EkfDogOdometryGPS3D()

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
