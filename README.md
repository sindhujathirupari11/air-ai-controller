🖐️ Air AI Controller (v2.0)

Air AI Controller is a Human-Computer Interaction (HCI) application that replaces the traditional mouse with a virtual, touchless interface.
It allows users to control their system and draw digitally using hand gestures captured through a webcam.

🛠️ Tech Stack
Language: Python (3.8 – 3.11 recommended)
Computer Vision: MediaPipe (BlazePalm Model)
Libraries: OpenCV, NumPy, PyAutoGUI

🚀 How It Works
1. Hand Tracking: Landmark Detection
Uses MediaPipe’s BlazePalm model
Detects 21 hand landmarks
Works reliably without depending on skin color or lighting

2. Signal Processing: Exponential Smoothing (EMA)
Raw webcam input can be noisy, causing cursor jitter.
To fix this, a smoothing filter is applied:
NewPosition = (α × Current) + ((1 - α) × Previous)
Implemented in utils.py
Provides stable and smooth cursor movement

3. Coordinate Mapping
Camera resolution: 1280 × 720
Screen resolution: 1920 × 1080
Solution:
Define an active region in camera frame
Map it to screen using numpy.interp
Result:
Full screen coverage
Less hand movement required

🎨 Features
Feature	Gesture	Description
Mouse Move	Index Finger Up	Move cursor smoothly
Left Click	Index + Thumb Pinch	Click when distance < 38px
Draw Mode	Index Finger Up	Draw with 8 colors
Hover Mode	Index + Middle Up	Move without drawing
Clear Canvas	Press C	Clears screen

✨ UI Features
Dark-themed interface
Real-time FPS display
Hand detection status
Glow effect for drawing

🏗️ Project Structure
air-ai-controller/
│
├── app.py                # Main application loop
├── hand_tracker.py       # Hand tracking logic
├── utils.py              # Smoothing and calculations
├── hand_landmarker.task  # Pre-trained model
├── requirements.txt
└── README.md

🚀 Future Scope
Velocity-based cursor control (dynamic speed)
Gesture-based scrolling

🛠️ Installation & Setup

1. Clone the Repository
git clone https://github.com/sindhujathirupari11/air-ai-controller.git
cd air-ai-controller

2. Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
python app.py

⚙️ System Requirements
Python 3.8 – 3.11
Webcam (720p or higher recommended)
OS: Windows (tested)

👩‍💻 Developed by
Sindhuja Thirupari
2nd Year B.Tech | Computer Science & Engineering

⭐ Support
If you found this project useful:
Star the repository
Fork and improve it