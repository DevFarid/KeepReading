import tkinter as tk
import threading  # Import the threading module
from CameraCV import CameraCV

class KeepReading:
    """
    Runs the GUI application.
    Created by Farid on 09/13/2023.
    """
    
    @staticmethod
    def run():
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
        capture_thread = threading.Thread(target=CameraCV.captureCameraPicture)
        capture_thread.start()

        # Run the Tkinter event loop
        root.mainloop()

KeepReading.run()
