# user_interface.py
import tkinter as tk
from PIL import Image, ImageTk

class UserInterface:
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.root = tk.Tk()
        self.root.title("Classroom AI Assistant")

        # Video display
        self.video_panel = tk.Label(self.root)
        self.video_panel.pack(side=tk.LEFT, padx=10, pady=10)

        # Summary and resources display
        self.summary_text = tk.Text(self.root, height=10, width=50)
        self.summary_text.pack(side=tk.TOP, padx=10, pady=10)

        self.resources_text = tk.Text(self.root, height=10, width=50)
        self.resources_text.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Control buttons
        self.start_button = tk.Button(self.root, text="Start", command=self.start_system)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.pause_button = tk.Button(self.root, text="Pause", command=self.pause_system)
        self.pause_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.format_button = tk.Button(self.root, text="Change Summary Format", command=self.change_summary_format)
        self.format_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.root.mainloop()

    def start_system(self):
        self.ai_system.run()

    def pause_system(self):
        self.ai_system.pause()

    def change_summary_format(self):
        self.ai_system.toggle_summary_format()

    def update_video_display(self, frame):
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.video_panel.configure(image=photo)
        self.video_panel.image = photo

    def update_summary_display(self, summary):
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)

    def update_resources_display(self, resources):
        self.resources_text.delete("1.0", tk.END)
        for resource in resources:
            self.resources_text.insert(tk.END, f"- {resource}\n")
