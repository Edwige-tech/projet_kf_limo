#!/usr/bin/env python3

from pathlib import Path
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


# ============================================================
# BAG PATH
# ============================================================

bags_dir = Path(
    "/home/edossou-yovo/Documents/kalman-ws/bags_data/bags_dog"
)

bag_path = max(
    [
        p for p in bags_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("rosbag2_")
    ],
    key=lambda p: p.stat().st_mtime
)


# ============================================================
# TOPICS
# ============================================================

TOPIC_EKF = (
    "/robot/localisation/ekf_position_stamped"
)

TOPIC_FILT = (
    "/robot/localisation/filtered_odom"
)

topic_types = {
    TOPIC_EKF: PoseStamped,
    TOPIC_FILT: Odometry
}


# ============================================================
# DATA
# ============================================================

ekf_data = []
filt_data = []


# ============================================================
# OPEN BAG
# ============================================================

reader = rosbag2_py.SequentialReader()

storage_options = rosbag2_py.StorageOptions(
    uri=str(bag_path),
    storage_id="sqlite3"
)

converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader.open(
    storage_options,
    converter_options
)

print(
    "Reading bag:",
    bag_path
)


# ============================================================
# READ MESSAGES
#
# IMPORTANT:
# use BAG timestamp for BOTH topics
# ============================================================

while reader.has_next():

    topic, data, timestamp = (
        reader.read_next()
    )

    if topic not in topic_types:
        continue

    msg = deserialize_message(
        data,
        topic_types[topic]
    )

    # --------------------------------------------------------
    # SAME TIME BASE FOR ALL DATA
    # --------------------------------------------------------

    t = timestamp * 1e-9

    # --------------------------------------------------------
    # EKF POSITION
    # --------------------------------------------------------

    if topic == TOPIC_EKF:

        x = float(
            msg.pose.position.x
        )

        y = float(
            msg.pose.position.y
        )

        ekf_data.append([
            t,
            x,
            y
        ])

    # --------------------------------------------------------
    # FILTERED ODOMETRY
    # --------------------------------------------------------

    elif topic == TOPIC_FILT:

        x = float(
            msg.pose.pose.position.x
        )

        y = float(
            msg.pose.pose.position.y
        )

        filt_data.append([
            t,
            x,
            y
        ])


# ============================================================
# NUMPY ARRAYS
# ============================================================

ekf_data = np.asarray(
    ekf_data,
    dtype=float
)

filt_data = np.asarray(
    filt_data,
    dtype=float
)


# ============================================================
# CHECK
# ============================================================

if ekf_data.size == 0:

    raise ValueError(
        f"No data found on {TOPIC_EKF}"
    )


if filt_data.size == 0:

    raise ValueError(
        f"No data found on {TOPIC_FILT}"
    )


# ============================================================
# SORT
# ============================================================

ekf_data = ekf_data[
    np.argsort(
        ekf_data[:, 0]
    )
]

filt_data = filt_data[
    np.argsort(
        filt_data[:, 0]
    )
]


# ============================================================
# SAVE
# ============================================================

np.save(
    "ekf_with_time.npy",
    ekf_data
)

np.save(
    "filt_with_time.npy",
    filt_data
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n===== SAVED FILES ====="
)

print(
    "ekf_with_time.npy:",
    ekf_data.shape
)

print(
    "filt_with_time.npy:",
    filt_data.shape
)


# ============================================================
# TIME INTERVALS
# ============================================================

print(
    "\n===== TIME INTERVALS ====="
)

print(
    "EKF:",
    ekf_data[0, 0],
    "->",
    ekf_data[-1, 0]
)

print(
    "Filtered odom:",
    filt_data[0, 0],
    "->",
    filt_data[-1, 0]
)


common_start = max(
    ekf_data[0, 0],
    filt_data[0, 0]
)

common_end = min(
    ekf_data[-1, 0],
    filt_data[-1, 0]
)


print(
    "\n===== COMMON INTERVAL ====="
)

print(
    common_start,
    "->",
    common_end
)


if common_start >= common_end:

    raise ValueError(
        "EKF and filtered odometry still do not "
        "share a common bag-time interval."
    )


print(
    "\nExtraction completed successfully."
)


"""
#!/usr/bin/env python3

from pathlib import Path
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


# BAG PATH

bags_dir = Path(
    "/home/edossou-yovo/Documents/kalman-ws/bags_data/bags_dog"
)

bag_path = max(
    [
        p for p in bags_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("rosbag2_")
    ],
    key=lambda p: p.stat().st_mtime
)


# TOPICS

TOPIC_EKF = (
    "/robot/localisation/ekf3d_position_stamped"
)

TOPIC_FILT = (
    "/robot/localisation/filtered_odom"
)

topic_types = {
    TOPIC_EKF: PoseStamped,
    TOPIC_FILT: Odometry
}


# DATA

ekf_data = []
filt_data = []


# OPEN BAG

reader = rosbag2_py.SequentialReader()

storage_options = rosbag2_py.StorageOptions(
    uri=str(bag_path),
    storage_id="sqlite3"
)

converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader.open(
    storage_options,
    converter_options
)

print(
    "Reading bag:",
    bag_path
)


# READ MESSAGES
# Use bag timestamp for both topics

while reader.has_next():

    topic, data, timestamp = (
        reader.read_next()
    )

    if topic not in topic_types:
        continue

    msg = deserialize_message(
        data,
        topic_types[topic]
    )

    # Same time base for all data
    t = timestamp * 1e-9

    # EKF 3D position
    if topic == TOPIC_EKF:

        x = float(
            msg.pose.position.x
        )

        y = float(
            msg.pose.position.y
        )

        ekf_data.append([
            t,
            x,
            y
        ])

    # Filtered odometry
    elif topic == TOPIC_FILT:

        x = float(
            msg.pose.pose.position.x
        )

        y = float(
            msg.pose.pose.position.y
        )

        filt_data.append([
            t,
            x,
            y
        ])


# NUMPY ARRAYS

ekf_data = np.asarray(
    ekf_data,
    dtype=float
)

filt_data = np.asarray(
    filt_data,
    dtype=float
)


# CHECK

if ekf_data.size == 0:

    raise ValueError(
        f"No data found on {TOPIC_EKF}"
    )


if filt_data.size == 0:

    raise ValueError(
        f"No data found on {TOPIC_FILT}"
    )


# SORT

ekf_data = ekf_data[
    np.argsort(
        ekf_data[:, 0]
    )
]

filt_data = filt_data[
    np.argsort(
        filt_data[:, 0]
    )
]


# SAVE

np.save(
    "ekf3d_with_time.npy",
    ekf_data
)

np.save(
    "filt3d_with_time.npy",
    filt_data
)


# SUMMARY

print(
    "\n===== SAVED FILES ====="
)

print(
    "ekf3d_with_time.npy:",
    ekf_data.shape
)

print(
    "filt3d_with_time.npy:",
    filt_data.shape
)


# TIME INTERVALS

print(
    "\n===== TIME INTERVALS ====="
)

print(
    "EKF 3D:",
    ekf_data[0, 0],
    "->",
    ekf_data[-1, 0]
)

print(
    "Filtered odom:",
    filt_data[0, 0],
    "->",
    filt_data[-1, 0]
)


common_start = max(
    ekf_data[0, 0],
    filt_data[0, 0]
)

common_end = min(
    ekf_data[-1, 0],
    filt_data[-1, 0]
)


print(
    "\n===== COMMON INTERVAL ====="
)

print(
    common_start,
    "->",
    common_end
)


if common_start >= common_end:

    raise ValueError(
        "EKF 3D and filtered odometry do not "
        "share a common bag-time interval."
    )


print(
    "\nEKF 3D extraction completed successfully."
)
"""
