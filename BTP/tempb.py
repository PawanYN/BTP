import cv2
import numpy as np

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

def compute_optical_flow(video_path, feature_vector_file_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize an empty list to store feature vectors
    feature_vectors = []

    while True:
        # Read the next frame
        ret, next_frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        # Compute the feature vector
        feature_vector = calculate_optical_flow(prev_gray, next_gray)
        # Append the feature vector to the list
        feature_vectors.append(feature_vector)
        # Update the previous frame and previous gray for the next iteration
        prev_gray = next_gray

    # Save the feature vectors to a file
    np.savetxt(feature_vector_file_path, np.array(feature_vectors), delimiter=',')
     
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'path/to/your/video.mp4'
feature_vector_file_path = 'path/to/save/feature_vectors.csv'
compute_optical_flow(video_path, feature_vector_file_path)
