import cv2
import mediapipe as mp
import numpy as np
import socketio
import tkinter as tk
from tkinter import ttk, messagebox

# --- GLOBAL VARIABLES ---
# These will hold the user's details after login
USER_DETAILS = {
    "name": "",
    "roll": "",
    "sap": ""
}

# --- GUI LOGIN CLASS ---
class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ProctorHQ | Student Login")
        self.root.geometry("400x350")
        self.root.configure(bg="#f8f9fa")
        
        # Center the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width/2) - (400/2)
        y = (screen_height/2) - (350/2)
        root.geometry('%dx%d+%d+%d' % (400, 350, x, y))

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background="#f8f9fa", font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'))

        # Header
        header_frame = tk.Frame(root, bg="#f8f9fa")
        header_frame.pack(pady=20)
        tk.Label(header_frame, text="Exam Portal", font=("Helvetica", 18, "bold"), bg="#f8f9fa", fg="#2c3e50").pack()
        tk.Label(header_frame, text="Enter your details to begin", font=("Helvetica", 10), bg="#f8f9fa", fg="#7f8c8d").pack()

        # Input Frame
        input_frame = tk.Frame(root, bg="#f8f9fa", padx=40)
        input_frame.pack(fill="both", expand=True)

        # Name
        ttk.Label(input_frame, text="Full Name:").pack(anchor="w", pady=(10,0))
        self.name_entry = ttk.Entry(input_frame, width=40)
        self.name_entry.pack(pady=5)

        # Roll No
        ttk.Label(input_frame, text="Roll Number:").pack(anchor="w", pady=(10,0))
        self.roll_entry = ttk.Entry(input_frame, width=40)
        self.roll_entry.pack(pady=5)

        # SAP ID
        ttk.Label(input_frame, text="SAP ID:").pack(anchor="w", pady=(10,0))
        self.sap_entry = ttk.Entry(input_frame, width=40)
        self.sap_entry.pack(pady=5)

        # Button
        self.login_btn = tk.Button(root, text="Start Exam", bg="#6366f1", fg="white", 
                                   font=("Helvetica", 11, "bold"), relief="flat", 
                                   padx=20, pady=8, cursor="hand2", command=self.submit)
        self.login_btn.pack(pady=20)

    def submit(self):
        name = self.name_entry.get().strip()
        roll = self.roll_entry.get().strip()
        sap = self.sap_entry.get().strip()

        if not name or not roll or not sap:
            messagebox.showerror("Error", "All fields are required!")
            return
        
        # Save details
        USER_DETAILS["name"] = name
        USER_DETAILS["roll"] = roll
        USER_DETAILS["sap"] = sap
        
        # Close Window
        self.root.destroy()

# --- RUN LOGIN SCREEN ---
root = tk.Tk()
app = LoginApp(root)
root.mainloop()

# If window closed without logging in (X button), stop script
if not USER_DETAILS["name"]:
    print("Login cancelled.")
    exit()

# ==========================================
#      BELOW IS THE EXISTING VISION LOGIC
# ==========================================

# --- NETWORK SETUP ---
sio = socketio.Client()
print(f"\nLogging in as {USER_DETAILS['name']}...")

try:
    sio.connect('http://localhost:3000')
    print("Connected to Proctor Server!")
    # Register student using the GUI data
    sio.emit('student-connect', USER_DETAILS)
except Exception as e:
    print(f"Connection Failed: {e}")
    print("Ensure server.js is running first!")
    exit()

# --- INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)

mp_drawing = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(0)

# --- CALIBRATION VARIABLES ---
pitch_offset = 0
yaw_offset = 0
roll_offset = 0
is_calibrated = False

# --- SMOOTHING VARIABLES ---
alpha = 0.2
smooth_pitch = 0
smooth_yaw = 0
smooth_roll = 0

# Track previous status
last_sent_status = ""

def get_head_pose(image, face_landmarks):
    img_h, img_w, img_c = image.shape
    face_3d = np.array([
        (0.0, 0.0, 0.0), (0.0, 330.0, -65.0), (-225.0, -170.0, -135.0),
        (225.0, -170.0, -135.0), (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)
    ], dtype=np.float64)

    face_2d = []
    target_indices = [1, 152, 33, 263, 61, 291]
    
    for idx in target_indices:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    face_2d = np.array(face_2d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
    dist_matrix = np.zeros((4,1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, img_c = image.shape

    fd_results = face_detection.process(image)
    fm_results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_count = 0
    status_text = "Status: Normal"
    color = (0, 255, 0)

    if fd_results.detections:
        face_count = len(fd_results.detections)
        for detection in fd_results.detections:
            mp_drawing.draw_detection(image, detection)

    cv2.putText(image, f"Faces: {face_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if not is_calibrated:
        cv2.putText(image, "Look at screen & Press 'C'", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if face_count == 0:
        status_text = "VIOLATION: NO FACE"
        color = (0, 0, 255)
    elif face_count > 1:
        status_text = "VIOLATION: MULTIPLE FACES"
        color = (0, 0, 255)
    elif face_count == 1:
        if fm_results.multi_face_landmarks:
            for face_landmarks in fm_results.multi_face_landmarks:
                raw_pitch, raw_yaw, raw_roll = get_head_pose(image, face_landmarks)
                smooth_pitch = (raw_pitch * alpha) + (smooth_pitch * (1.0 - alpha))
                smooth_yaw = (raw_yaw * alpha) + (smooth_yaw * (1.0 - alpha))
                smooth_roll = (raw_roll * alpha) + (smooth_roll * (1.0 - alpha))

                final_pitch = smooth_pitch - pitch_offset
                final_yaw = smooth_yaw - yaw_offset

                if abs(final_pitch) > 25:
                    status_text = "VIOLATION: LOOKING AWAY"
                    color = (0, 0, 255)
                elif abs(final_yaw) > 40:
                    status_text = "VIOLATION: SIDEWAYS LOOK"
                    color = (0, 0, 255)
        else:
            status_text = "VIOLATION: FACE NOT CLEAR"
            color = (0, 0, 255)

    if is_calibrated:
        cv2.putText(image, status_text, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if status_text != last_sent_status:
            sio.emit('student-status-update', status_text)
            last_sent_status = status_text

    cv2.imshow('Proctoring Vision Core', image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        pitch_offset = smooth_pitch
        yaw_offset = smooth_yaw
        roll_offset = smooth_roll
        is_calibrated = True

cap.release()
cv2.destroyAllWindows()
sio.disconnect()