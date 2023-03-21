"""
https://github.com/Yaga987/Games/blob/main/game2.py
"""
import cv2
import numpy as np
import random

# Set up the game window
window_size = (640, 480)
window_name = 'Color Catcher'
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, 0, 0)

# Set up the game objects
catcher_color = (0, 0, 255)
catcher_width = 100
catcher_height = 20
catcher_position = np.array([window_size[0]//2, window_size[1]-catcher_height], dtype=int)
catcher_velocity = np.array([10, 0], dtype=int)

ball_radius = 20
ball_speed = 5
ball_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
balls = []
score = 0
# Set up the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale and apply a threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area, which should be the hand
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        hand_hull = cv2.convexHull(hand_contour)
        hand_center = np.mean(hand_contour, axis=0, dtype=int)[0]
        
        # Move the catcher based on the position of the hand
        if hand_center[0] < catcher_position[0] and catcher_position[0] > 0:
            catcher_position -= catcher_velocity
        elif hand_center[0] > catcher_position[0]+catcher_width and catcher_position[0]+catcher_width < window_size[0]:
            catcher_position += catcher_velocity
    
    # Generate a new ball with a random color and position if necessary
    if len(balls) < 5:
        ball_color = random.choice(ball_colors)
        ball_position = np.array([random.randint(ball_radius, window_size[0]-ball_radius), 0], dtype=int)
        ball_velocity = np.array([0, ball_speed], dtype=int)
        balls.append((ball_color, ball_position, ball_velocity))
    
    # Move the balls and check for collisions with the catcher
    for i in range(len(balls)):
        balls[i] = (balls[i][0], balls[i][1]+balls[i][2], balls[i][2])
        ball_position = balls[i][1]
        if ball_position[1]+ball_radius >= catcher_position[1] and \
           ball_position[0] >= catcher_position[0] and \
           ball_position[0] <= catcher_position[0]+catcher_width:
            balls.pop(i)
            score += 1
            break
        elif ball_position[1]+ball_radius >= window_size[1]:
            balls.pop(i)
            break
    
    # Draw the game objects on the frame
    frame = np.zeros(window_size+(3,), dtype=np.uint8)
    cv2.rectangle(frame, tuple(catcher_position), tuple(catcher_position+np.array([catcher_width, catcher_height])), catcher_color, -1)
    for ball in balls:
        cv2.circle(frame, tuple(ball[1]), ball_radius, ball[0], -1)
    
    # Display the frame
    cv2.putText(frame, "Score: {}".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, frame)
    
    # Exit the game if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the game window
cap.release()
cv2.destroyAllWindows()