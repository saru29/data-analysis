import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from ttkthemes import ThemedStyle

def switch_to_stretches():
    # Code to switch to stretches mode
    print("Switched to stretches mode")
    btn_stretches.configure(style="SelectedIconButton.TButton")  # Highlight the button for stretches
    btn_cycling.configure(style="IconButton.TButton")  # Reset the button for cycling
    btn_cooldowns.configure(style="IconButton.TButton")  # Reset the button for cooldowns

def switch_to_cycling():
    # Code to switch to cycling mode
    print("Switched to cycling mode")
    btn_stretches.configure(style="IconButton.TButton")  # Reset the button for stretches
    btn_cycling.configure(style="SelectedIconButton.TButton")  # Highlight the button for cycling
    btn_cooldowns.configure(style="IconButton.TButton")  # Reset the button for cooldowns

def switch_to_cooldowns():
    # Code to switch to cooldowns mode
    print("Switched to cooldowns mode")
    btn_stretches.configure(style="IconButton.TButton")  # Reset the button for stretches
    btn_cycling.configure(style="IconButton.TButton")  # Reset the button for cycling
    btn_cooldowns.configure(style="SelectedIconButton.TButton")  # Highlight the button for cooldowns

# Create the main application window
window = tk.Tk()
window.title("Workout Modes")
window.config(bg="white")

# Create a themed style
style = ThemedStyle(window)
style.set_theme("arc")  # Set the theme to "arc" for rounded buttons

# Define the custom button styles
style.configure("IconButton.TButton",
                background="white",
                relief="flat",
                borderwidth=0)
style.configure("SelectedIconButton.TButton",
                background="white",
                relief="solid",
                borderwidth=5,
                bordercolor="red",
                borderradius=8)  # Adjust the border radius as desired

# Create white icons for the buttons
stretches_icon = ImageTk.PhotoImage(Image.open("warmup.png").resize((32, 32)).convert("RGBA"))
cycling_icon = ImageTk.PhotoImage(Image.open("cycling.png").resize((32, 32)).convert("RGBA"))
cooldowns_icon = ImageTk.PhotoImage(Image.open("cooldown.png").resize((32, 32)).convert("RGBA"))

# Create buttons with icons for mode selection
btn_stretches = ttk.Button(window, image=stretches_icon, command=switch_to_stretches, style="IconButton.TButton")
btn_stretches.pack(pady=10)

btn_cycling = ttk.Button(window, image=cycling_icon, command=switch_to_cycling, style="IconButton.TButton")
btn_cycling.pack(pady=10)

btn_cooldowns = ttk.Button(window, image=cooldowns_icon, command=switch_to_cooldowns, style="IconButton.TButton")
btn_cooldowns.pack(pady=10)

# Initially highlight the button for stretches as the default mode
btn_stretches.configure(style="SelectedIconButton.TButton")

# Run the main event loop
window.mainloop()