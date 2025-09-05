
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# BODY_25 keypoint connections
BODY_25_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20),
    (11, 22), (22, 23)
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_joint_coords(keypoints):
    return [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]

def visualize_skeleton(joints, output_path=None):
    xs = [x for x, y, c in joints]
    ys = [y for x, y, c in joints]
    confs = [c for x, y, c in joints]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c='red', s=10)
    for i, j in BODY_25_PAIRS:
        if confs[i] > 0.1 and confs[j] > 0.1:
            plt.plot([xs[i], xs[j]], [ys[i], ys[j]], 'b-', linewidth=1)
    plt.gca().invert_yaxis()
    plt.axis('off')
    if output_path:
        plt.savefig(output_path)
    plt.close()

def compute_angles(joints):
    head = joints[0][:2]
    neck = joints[1][:2]
    mid_hip = joints[8][:2]
    right_shoulder = joints[2][:2]
    right_hip = joints[9][:2]
    right_knee = joints[10][:2]
    right_ankle = joints[11][:2]

    angles = {}
    if all(j[2] > 0.1 for j in [joints[0], joints[1], joints[8]]):
        angles['neck_flexion'] = calculate_angle(head, neck, mid_hip)
    if all(j[2] > 0.1 for j in [joints[2], joints[9], joints[10]]):
        angles['trunk_lean'] = calculate_angle(right_shoulder, right_hip, right_knee)
    if all(j[2] > 0.1 for j in [joints[9], joints[10], joints[11]]):
        angles['knee_flexion'] = calculate_angle(right_hip, right_knee, right_ankle)
    return angles

# 🔧 Set your folder path here
json_folder = r'C:\Users\jacander\Desktop\OpenPose\openpose-master\output_json'  # Update this path

# 📄 Output CSV file
csv_output = "joint_angles_all_files.csv"
with open(csv_output, 'w', newline='') as csvfile:
    fieldnames = ['file_name', 'person_index', 'neck_flexion', 'trunk_lean', 'knee_flexion']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                for idx, person in enumerate(data.get('people', [])):
                    keypoints = person.get('pose_keypoints_2d', [])
                    joints = get_joint_coords(keypoints)
                    angles = compute_angles(joints)
                    angles['file_name'] = filename
                    angles['person_index'] = idx
                    writer.writerow(angles)
                    # Optional: save skeleton visualization
                    # visualize_skeleton(joints, output_path=f"skeleton_{filename}_{idx}.png")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print(f" Joint angles from all JSON files saved to {csv_output}")
