import cv2
import mediapipe as mp
import numpy as np
import socketio
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk 

# --- GLOBAL VARIABLES ---
USER_DETAILS = {
    "name": "",
    "roll": "",
    "sap": ""
}

class ProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ProctorHQ Secure Client")
        self.root.geometry("1000x800") # Increased size
        self.root.resizable(True, True) # Allow resizing
        
        # COLORS
        self.colors = {
            "primary": "#6366f1",    # Indigo
            "bg": "#ffffff",         # White
            "text": "#1e293b",       # Slate 800
            "danger": "#dc2626"      # Red
        }
        
        # NETWORK & VISION SETUP
        self.sio = socketio.Client()
        self.setup_vision()
        
        # STATE VARIABLES
        self.is_exam_running = False
        self.pitch_offset = 0; self.yaw_offset = 0; self.is_calibrated = False
        self.smooth_pitch = 0; self.smooth_yaw = 0
        self.last_sent_status = ""
        self.cap = None

        # UI SETUP
        self.setup_styles()
        self.build_login_ui()

    def setup_vision(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.mp_drawing = mp.solutions.drawing_utils 

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure("Modern.TButton", background=self.colors["primary"], foreground="white", font=("Segoe UI", 11, "bold"), borderwidth=0)
        style.map("Modern.TButton", background=[("active", "#4f46e5")])
        
        # Danger Button Style (Red)
        style.configure("Danger.TButton", background=self.colors["danger"], foreground="white", font=("Segoe UI", 11, "bold"), borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#b91c1c")])

    # ==========================
    #      UI: LOGIN SCREEN
    # ==========================
    def build_login_ui(self):
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Header
        header = tk.Frame(self.root, bg=self.colors["primary"], height=100)
        header.pack(fill="x")
        tk.Label(header, text="ProctorHQ Login", bg=self.colors["primary"], fg="white", font=("Segoe UI", 24, "bold")).place(relx=0.5, rely=0.5, anchor="center")

        # Container
        frame = ttk.Frame(self.root, padding=40)
        frame.pack(expand=True)

        # Inputs
        self.create_input(frame, "Full Name", "name_entry")
        self.create_input(frame, "Roll Number", "roll_entry")
        self.create_input(frame, "SAP ID", "sap_entry")

        # CHECKBOX (New Feature)
        self.agree_var = tk.IntVar()
        cb = tk.Checkbutton(frame, text="I ensure to give the exam fairly and adhere to all integrity guidelines.", 
                            variable=self.agree_var, bg=self.colors["bg"], activebackground=self.colors["bg"], font=("Segoe UI", 9))
        cb.pack(pady=(20, 10))

        # Submit
        ttk.Button(frame, text="Start Exam", style="Modern.TButton", command=self.attempt_login).pack(fill="x", pady=20, ipady=5)

    def create_input(self, parent, label, var_name):
        ttk.Label(parent, text=label, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 5))
        entry = ttk.Entry(parent, width=40, font=("Segoe UI", 11))
        entry.pack(fill="x")
        setattr(self, var_name, entry)

    def attempt_login(self):
        name = self.name_entry.get().strip()
        roll = self.roll_entry.get().strip()
        sap = self.sap_entry.get().strip()

        if not name or not roll or not sap:
            messagebox.showwarning("Error", "Please fill all fields.")
            return

        if self.agree_var.get() == 0:
            messagebox.showwarning("Compliance", "You must agree to the fair exam terms.")
            return

        # Save & Connect
        USER_DETAILS.update({"name": name, "roll": roll, "sap": sap})
        
        try:
            self.sio.connect('http://localhost:3000')
            self.sio.emit('student-connect', USER_DETAILS)
            self.start_exam_mode()
        except Exception as e:
            messagebox.showerror("Connection Error", f"Is server.js running?\nError: {e}")

    # ==========================
    #      UI: EXAM SCREEN (FIXED)
    # ==========================
    def start_exam_mode(self):
        # Switch UI
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.configure(bg="black") # Dark mode for exam focus

        self.cap = cv2.VideoCapture(0)
        self.is_exam_running = True

        # --- LAYOUT FIX: Pack Bottom Bar FIRST ---
        
        # 1. Controls Bar (Bottom)
        controls = tk.Frame(self.root, bg="white", height=80)
        controls.pack(side=tk.BOTTOM, fill=tk.X)
        controls.pack_propagate(False) # Force height

        # Status Label (Left)
        self.status_lbl = tk.Label(controls, text="Status: Calibrating...", font=("Segoe UI", 14, "bold"), fg="orange", bg="white")
        self.status_lbl.pack(side="left", padx=20)

        # End Exam Button (Right)
        btn_end = ttk.Button(controls, text="END EXAM", style="Danger.TButton", command=self.end_exam)
        btn_end.pack(side="right", padx=20, ipadx=10, ipady=5)

        # 2. Video Frame (Fills Remaining Space)
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Start Loop
        self.process_video_loop()

    # ==========================
    #      LOGIC: VISION CORE
    # ==========================
    def process_video_loop(self):
        if not self.is_exam_running:
            return

        success, frame = self.cap.read()
        if success:
            # 1. Process Logic
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, status, color = self.analyze_frame(frame)

            # 2. Resize Image to Fit Window
            # Get current window dimensions
            win_w = self.root.winfo_width()
            win_h = self.root.winfo_height() - 80 # Subtract control bar height
            
            # Convert to PIL Image
            img = Image.fromarray(processed_frame)
            
            # Intelligent Resize (Maintain Aspect Ratio)
            if win_w > 1 and win_h > 1: # Avoid startup errors
                img_ratio = img.width / img.height
                win_ratio = win_w / win_h
                
                if win_ratio > img_ratio:
                    new_h = win_h
                    new_w = int(new_h * img_ratio)
                else:
                    new_w = win_w
                    new_h = int(new_w / img_ratio)
                    
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # 3. Update Status Text
            self.status_lbl.config(text=status, fg=color)
            
            # 4. Network Sync
            if self.is_calibrated and status != self.last_sent_status:
                self.sio.emit('student-status-update', status)
                self.last_sent_status = status

        # Loop every 20ms (lower CPU usage)
        self.root.after(20, self.process_video_loop)

    def analyze_frame(self, image):
        img_h, img_w, _ = image.shape
        status_text = "Status: Normal"
        tk_color = "green" 
        
        fd_results = self.mp_face_detection.process(image)
        fm_results = self.mp_face_mesh.process(image)
        
        face_count = 0
        if fd_results.detections:
            face_count = len(fd_results.detections)
            for detection in fd_results.detections:
                self.mp_drawing.draw_detection(image, detection)

        # Draw Calibration Prompt on Video
        if not self.is_calibrated:
            cv2.putText(image, "Look at screen & Press 'C'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            status_text = "Action Required: Press 'C' to Calibrate"
            tk_color = "#f59e0b" # Orange
        
        # Violation Logic
        if face_count == 0:
            status_text = "VIOLATION: NO FACE"
            tk_color = "#dc2626" # Red
        elif face_count > 1:
            status_text = "VIOLATION: MULTIPLE FACES"
            tk_color = "#dc2626"
        elif face_count == 1 and fm_results.multi_face_landmarks:
            for face_landmarks in fm_results.multi_face_landmarks:
                pitch, yaw, _ = self.get_head_pose(image, face_landmarks, img_w, img_h)
                
                alpha = 0.2
                self.smooth_pitch = (pitch * alpha) + (self.smooth_pitch * (1.0 - alpha))
                self.smooth_yaw = (yaw * alpha) + (self.smooth_yaw * (1.0 - alpha))

                if self.is_calibrated:
                    final_pitch = self.smooth_pitch - self.pitch_offset
                    final_yaw = self.smooth_yaw - self.yaw_offset

                    if abs(final_pitch) > 25:
                        status_text = "VIOLATION: LOOKING AWAY"
                        tk_color = "#dc2626"
                    elif abs(final_yaw) > 40:
                        status_text = "VIOLATION: SIDEWAYS LOOK"
                        tk_color = "#dc2626"

        return image, status_text, tk_color

    def get_head_pose(self, image, face_landmarks, img_w, img_h):
        face_3d = np.array([
            (0.0, 0.0, 0.0), (0.0, 330.0, -65.0), (-225.0, -170.0, -135.0),
            (225.0, -170.0, -135.0), (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)
        ], dtype=np.float64)
        face_2d = []
        for idx in [1, 152, 33, 263, 61, 291]:
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
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0], angles[1], angles[2]

    def calibrate(self, event=None):
        if self.is_exam_running:
            self.pitch_offset = self.smooth_pitch
            self.yaw_offset = self.smooth_yaw
            self.is_calibrated = True

    def end_exam(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to submit and end the exam?"):
            self.is_exam_running = False
            if self.cap: self.cap.release()
            self.sio.disconnect()
            self.root.quit()

# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ProctorApp(root)
    # Bind 'c' key to calibration function
    root.bind('<c>', app.calibrate)
    root.bind('<C>', app.calibrate)
    root.mainloop()