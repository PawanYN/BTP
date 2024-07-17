import cv2
import numpy as np
from google.colab.patches import cv2_imshow


def calculate_optical_flow(prev_gray, next_gray):

    # Calculate optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize the angle by dividing by 2*pi to scale it between 0 and 1
    angle /= (2 * np.pi)

    # Flatten and concatenate magnitude and angle to form the feature vector
    feature_vector = np.hstack((magnitude.flatten(), angle.flatten()))

    return feature_vector


def compute_optical_flow(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize an empty list to store feature vectors
    feature_vectors = []

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask for visualization
    mask = np.zeros_like(prev_frame)
    mask[..., 1] = 255  # Make the mask green

    while True:
        # Read the next frame
        ret, next_frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Compute the optical flow using the Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Set the mask according to the flow magnitude and angle
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to RGB
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        feature_vector=calculate_optical_flow(prev_gray, next_gray)
        # Display the result
        print('original image')
        cv2_imshow(next_frame)
        print('original optical flow representaion')
        cv2_imshow(rgb)
        print(feature_vector)
        print(feature_vector.shape)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update the previous frame and previous gray
        prev_gray = next_gray

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = '/content/drive/MyDrive/BTP24/YOLOV8/frame_output/cropped_video_yolov8_C0016_2.mp4'
compute_optical_flow(video_path)

