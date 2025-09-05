import cv2

# load the video
video_path = r"\\medctr\dfs\cib$\shared\02_projects\mouthpiece_data_collection\Football\00_2023_Football_Technique\02_VideoClips\Combined\08.01.2023 Mp1042 Imp8_comp.m4v"  #change this
cap = cv2.VideoCapture(video_path)

# check if successful opening
if not cap.isOpened():
    print("Error: Could not open video")
    exit()
    
# read first frame to get dimensions
ret, frame = cap.read()
if not ret: 
    print("Error: Could not read frame")
    exit()

# get frame dimensions
height, width, _ = frame.shape
print(f"Original video dimensions: {width}x{height}")

# define crop size
crop_width = 100
crop_height = 500

# calculate top-left corner of crop box (centered)
x_start = (width - crop_width) // 2
y_start = (height - crop_height - 150) // 2

# crop the frame
cropped_frame = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]

# Show the cropped frame
cv2.imshow("Cropped Frame", cropped_frame)
cv2.waitKey(0)
cv2.destroyAllWindows(1)

# Reset the video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Define output video settings
output_path = r"\\medctr\dfs\cib$\shared\02_projects\mouthpiece_data_collection\Football\00_2023_Football_Technique\02_VideoClips\OpenPose Testing\08.01.2023 Mp1042 Imp8_comp_cropped.m4v" # change this
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = cap.get(cv2.CAP_PROP_FPS)           # Frame rate
out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

# Loop through all frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame
    cropped_frame = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]

    # Write the cropped frame to the output video
    out.write(cropped_frame)

# Release everything
cap.release()
out.release()
print(f" Cropped video saved to: {output_path}")
