import tkinter as tk
import cv2
from PIL import Image, ImageTk
import threading  # Import the threading module
import pytesseract
from CameraCV import CameraCV

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class KeepReading:

    def __init__(self) -> None:
        self.cv = CameraCV()
        self.run()

    """
    Runs the GUI application.
    Created by Farid on 09/13/2023.
    """
    
    def run(self):
        """
        Start the application.
        """        
        # Initialize Tkinter
        root = tk.Tk()
        root.title("KeepReading")
        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calculate width and height based on a 4:3 aspect ratio
        # Here we take 80% of the screen size to make sure the window fits in the screen
        width = int(screen_width * 0.8)
        height = int(width * 3 / 4)  # 4:3 aspect ratio

        # If calculated height is greater than screen height, recalculate dimensions
        if height > screen_height:
            height = int(screen_height * 0.8)
            width = int(height * 4 / 3)

        # Position the window in the center
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        # Set up the Tkinter window
        root.geometry(f"{width}x{height}+{x}+{y}")

        # Create a new thread to run the long-running task
        # capture_thread = threading.Thread(target=CameraCV.captureCameraPicture)
        # capture_thread.start()

        label_widget = tk.Label(root)
        label_widget.pack()

        # Event loop for camera preview
        def run_preview():
            img_update = self.cv.captureCameraPreview(400, 300)
            label_widget.configure(image=img_update)
            label_widget.image=img_update
            label_widget.update()
            label_widget.after(10, run_preview)
            
        button1 = tk.Button(root, text="Snap Picture", command=self.cv.analyze)
        button1.pack()

        run_preview()
        # Run the Tkinter event loop
        root.mainloop()

KeepReading()
