
from pathlib import Path
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


# Bag path
# --------------------------------------------------

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


# Topics
# --------------------------------------------------

TOPIC_FILT = "/robot/localisation/filtered_odom"
# TOPIC_EST = "/dog_kalman/estimated_position_stamped"
TOPIC_EST = "/robot/localisation/estimated_position_stamped"

topic_types = {
    TOPIC_EST: PoseStamped,
    TOPIC_FILT: Odometry
}

est_data = []
filt_data = []


# Open rosbag
# --------------------------------------------------

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

print("Reading bag:", bag_path)


# Read messages
# --------------------------------------------------

while reader.has_next():

    topic, data, timestamp = reader.read_next()

    if topic not in topic_types:
        continue

    msg = deserialize_message(
        data,
        topic_types[topic]
    )

    # Same bag time for both signals

    t = timestamp * 1e-9

    # Estimated position

    if topic == TOPIC_EST:

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)

        est_data.append([
            t,
            x,
            y
        ])

    # Filtered odometry

    elif topic == TOPIC_FILT:

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        filt_data.append([
            t,
            x,
            y
        ])


# Convert to numpy arrays
# --------------------------------------------------

est_data = np.asarray(est_data, dtype=float)
filt_data = np.asarray(filt_data, dtype=float)


# Check extracted data
# --------------------------------------------------

if est_data.size == 0:
    raise ValueError(
        f"No data found on {TOPIC_EST}"
    )

if filt_data.size == 0:
    raise ValueError(
        f"No data found on {TOPIC_FILT}"
    )


# Sort by timestamp
# --------------------------------------------------

est_data = est_data[
    np.argsort(est_data[:, 0])
]

filt_data = filt_data[
    np.argsort(filt_data[:, 0])
]


# Save
# --------------------------------------------------

np.save(
    "est_with_time.npy",
    est_data
)

np.save(
    "filt_with_time.npy",
    filt_data
)


# Summary
# --------------------------------------------------

print("\n===== SAVED FILES =====")

print(
    "est_with_time.npy :",
    est_data.shape
)

print(
    "filt_with_time.npy :",
    filt_data.shape
)

print("\n===== TIME INTERVALS =====")

print(
    "Estimated position:",
    est_data[0, 0],
    "->",
    est_data[-1, 0]
)

print(
    "Filtered odometry:",
    filt_data[0, 0],
    "->",
    filt_data[-1, 0]
)

common_start = max(
    est_data[0, 0],
    filt_data[0, 0]
)

common_end = min(
    est_data[-1, 0],
    filt_data[-1, 0]
)

print("\n===== COMMON INTERVAL =====")

print(
    common_start,
    "->",
    common_end
)

if common_start >= common_end:
    raise ValueError(
        "Estimated position and filtered odometry "
        "do not share a common time interval."
    )

print("\nExtraction completed successfully.")
