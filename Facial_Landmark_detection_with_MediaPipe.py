import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Read the image
img_base = cv2.imread('Images/IMG_5724.JPG')

# Make a copy of the image
img = img_base.copy()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Process the image to detect face landmarks
results = face_mesh.process(img)

# Check if face landmarks are detected
if results.multi_face_landmarks:
    # Iterate over each face
    for face_landmarks in results.multi_face_landmarks:
        # Draw the landmarks on the image
        for landmark in face_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])

            # Draw a circle at each landmark point
            # cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

    # Display the image with face landmarks
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(img[:,:,::-1])
    # plt.axis('off')
    # plt.show()

    #print(mp_face_mesh.FACEMESH_LEFT_EYE)

# Draw lines between left eye landmarks
for face_landmarks in results.multi_face_landmarks:
    for source_idx, target_idx in mp_face_mesh.FACEMESH_TESSELATION:
        source = face_landmarks.landmark[source_idx]
        target = face_landmarks.landmark[target_idx]

        relative_source = (int(source.x * img.shape[1]), int(source.y * img.shape[0]))
        relative_target = (int(target.x * img.shape[1]), int(target.y * img.shape[0]))
        cv2.line(img, relative_source, relative_target, (0, 255, 125), thickness=2)

# Display the image with lines drawn between left eye landmarks
fig = plt.figure(figsize=(10, 10))
plt.imshow(img[:, :, ::-1])
plt.axis('off')
plt.show()

# Release resources
face_mesh.close()
