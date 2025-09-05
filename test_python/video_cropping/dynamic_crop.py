import cv2          # OpenCV library for video and image processing
import os           # For file and folder operations
import glob         # For finding files using patterns (e.g., *.m4v)

# Set the input folder path (where your original videos are)
input_folder = r"\\medctr\dfs\cib$\shared\02_projects\mouthpiece_data_collection\Football\00_2023_Football_Technique\02_VideoClips\OpenPose Testing"

# Create a subfolder named 'Cropped' inside the input folder to save cropped videos
output_folder = os.path.join(input_folder, "Cropped")

# Find all .m4v video files in the input folder
video_files = glob.glob(os.path.join(input_folder, "*.m4v"))

# Variables to track mouse drawing state and coordinates
drawing = False
ix, iy = -1, -1  # Initial mouse click coordinates
fx, fy = -1, -1  # Final mouse release coordinates
crop_box = None  # Will store the final crop rectangle (x, y, width, height)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, crop_box, preview_frame

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # While dragging
        preview = preview_frame.copy()
        cv2.rectangle(preview, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Draw Crop", preview)

    elif event == cv2.EVENT_LBUTTONUP:  # Finish drawing
        drawing = False
        fx, fy = x, y
        crop_box = (min(ix, fx), min(iy, fy), abs(fx - ix), abs(fy - iy))  # Normalize box
        preview = preview_frame.copy()
        cv2.rectangle(preview, (ix, iy), (fx, fy), (0, 255, 0), 2)
        cv2.imshow("Draw Crop", preview)

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)  # Open the video
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        continue

    ret, frame = cap.read()  # Read the first frame
    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        cap.release()
        continue

# Show first frame and let user drop crop
    preview_frame = frame.copy()
    crop_box = None
    cv2.namedWindow("Draw Crop")
    cv2.setMouseCallback("Draw Crop", draw_rectangle)
    cv2.imshow("Draw Crop", preview_frame)

    print(f"Drawing crop for: {os.path.basename(video_path)}")
    print("Press 'c' to confirm crop, 'r' to reset, or 's' to skip this video.")

# Wait for user input
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c') and crop_box: # confirm crop
            break
        elif key == ord('r'): # reset crop
            crop_box = None
            preview_frame = frame.copy()
            cv2.imshow("Draw Crop", preview_frame)
        elif key == ord('s'): # skip video
            cap.release()
            cv2.destroyAllWindows()
            print(f"Skipped: {os.path.basename(video_path)}")
            break

    if not crop_box:
        continue # skip if no crop confirmed

# Crop and save video
    x, y, w, h = crop_box
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # go back to beginning of video
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# Process all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[y:y+h, x:x+w]
        out.write(cropped_frame)

# Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Cropped video saved to: {output_path}")
