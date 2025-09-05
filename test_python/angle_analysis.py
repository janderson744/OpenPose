
import os
import json
import csv
import numpy as np


# Step 1: Define the folder containing JSON files
# This should be the folder where OpenPose saved the output for one video
json_folder = "output_json/08.08.2023_Mp1047_Imp11_Mp1066_Imp6_comp"


# Step 2: Define the output CSV file name
# This is where joint angles will be saved
csv_output = "joint_angles_output.csv"



# Step 3: Convert flat keypoint list into (x, y, confidence) tuples
def get_joint_coords(keypoints):
    # OpenPose gives keypoints as a flat list: [x0, y0, c0, x1, y1, c1, ...]
    # This splits them into groups of 3: (x, y, confidence)
    return [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]



# Step 4: Calculate angle at joint 'b' using points a-b-c
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)  # Convert to NumPy arrays
    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)  # Avoid divide-by-zero
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to valid range
    return np.degrees(angle)  # Convert radians to degrees



# Step 5: Compute joint angles from BODY_25 keypoints
def compute_angles(joints):
    angles = {}  # Dictionary to store angles

    try:
        # Extract key joint positions (x, y only)
        head = joints[0][:2]
        neck = joints[1][:2]
        mid_hip = joints[8][:2]
        right_shoulder = joints[2][:2]
        right_hip = joints[9][:2]
        right_knee = joints[10][:2]
        right_ankle = joints[11][:2]

        # Only compute angles if confidence scores are high enough (> 0.1)
        if all(j[2] > 0.1 for j in [joints[0], joints[1], joints[8]]):
            angles['neck_flexion'] = calculate_angle(head, neck, mid_hip)
        if all(j[2] > 0.1 for j in [joints[2], joints[9], joints[10]]):
            angles['trunk_lean'] = calculate_angle(right_shoulder, right_hip, right_knee)
        if all(j[2] > 0.1 for j in [joints[9], joints[10], joints[11]]):
            angles['knee_flexion'] = calculate_angle(right_hip, right_knee, right_ankle)
    except IndexError:
        pass  # If keypoints are missing, skip this frame

    return angles



# Step 6: Open CSV file for writing results
with open(csv_output, 'w', newline='') as csvfile:
    fieldnames = ['frame', 'person_index', 'neck_flexion', 'trunk_lean', 'knee_flexion']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Write column headers



    # Step 7: Loop through all JSON files in the folder
    for filename in sorted(os.listdir(json_folder)):
        if filename.endswith('.json'):  # Only process JSON files
            frame_id = filename.split('_')[-2]  # Extract frame number from filename
            file_path = os.path.join(json_folder, filename)

            with open(file_path, 'r') as f:
                data = json.load(f)  # Load JSON content

            # 👤 Loop through each person detected in the frame
            for idx, person in enumerate(data.get('people', [])):
                keypoints = person.get('pose_keypoints_2d', [])  # Get BODY_25 keypoints
                joints = get_joint_coords(keypoints)  # Convert to (x, y, confidence)
                angles = compute_angles(joints)  # Compute joint angles
                angles['frame'] = frame_id  # Add frame number
                angles['person_index'] = idx  # Add person index
                writer.writerow(angles)  # Save to CSV


