# ui_utils.py
# Contains utility functions for enhanced UI components

import tkinter as tk
from tkinter import ttk, Frame, Label, Button
from PIL import Image, ImageDraw, ImageFont, ImageTk
import time
import os

# Modern color palette
UI_COLORS = {
    "primary": "#31BD7E",      # Green
    "secondary": "#5F5BE9",    # Blue
    "background": "#f0f0f0",   # Light gray
    "text": "#333333",         # Dark gray
    "disabled": "#a5a3a3",     # Light gray
    "button_text": "#FFFFFF",  # White
    "hover_primary": "#6A2399", # Purple
    "hover_secondary": "#F57C00", # Darker orange for hover
    "error": "#EB190A",        # Red
    "success": "#4CAF50",      # Green
    "warning": "#FFC107",      # Amber
    "info": "#2196F3"          # Light Blue
}

# Font configurations - using system-compatible fonts
UI_FONTS = {
    "title": ("Arial", 32, "bold"),
    "subtitle": ("Arial", 24, "bold"),
    "heading": ("Arial", 18, "bold"),
    "subheading": ("Arial", 16, "bold"),
    "body": ("Arial", 14),
    "body_bold": ("Arial", 14, "bold"),
    "small": ("Arial", 12),
    "small_bold": ("Arial", 12, "bold"),
    "tiny": ("Arial", 10),
    "button": ("Arial", 14, "bold")
}

# Keep references to prevent garbage collection
button_images = {}

def create_rounded_button(width=200, height=50, radius=15, bg_color="#5B79ED", text="Button", 
                         font=UI_FONTS["button"], text_color="#FFFFFF", key=None):
    """
    Create a rounded rectangle button image with text.
    
    Args:
        width (int): Button width
        height (int): Button height
        radius (int): Corner radius
        bg_color (str): Background color in hex format
        text (str): Button text
        font (tuple): Font configuration (family, size, style)
        text_color (str): Text color in hex format
        key (str): Unique key for this button to prevent garbage collection
        
    Returns:
        ImageTk.PhotoImage: Button image
    """
    # Create blank image with transparency
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=bg_color)
    
    # Add text
    try:
        # Try to use the specified font
        try:
            font_obj = ImageFont.truetype(font[0], font[1])
        except (OSError, IOError):
            # Try with a system font
            font_obj = ImageFont.truetype("arial.ttf", font[1])
    except (OSError, IOError):
        # Fallback to default font
        font_obj = ImageFont.load_default()
    
    # Get text size
    text_width, text_height = draw.textbbox((0, 0), text, font=font_obj)[2:4]
    
    # Center text
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    
    # Draw text
    draw.text((text_x, text_y), text, fill=text_color, font=font_obj)
    
    # Convert to PhotoImage
    button_img = ImageTk.PhotoImage(img)
    
    # Store reference to prevent garbage collection
    if key:
        button_images[key] = button_img
    
    return button_img

def create_styled_button(parent, text, command, width=200, height=50, bg_color=UI_COLORS["primary"], 
                        hover_color=UI_COLORS["hover_primary"], text_color=UI_COLORS["button_text"], 
                        font=UI_FONTS["button"], radius=15, key=None, state=tk.NORMAL):
    """
    Create a styled button with rounded corners and hover effect.
    
    Args:
        parent: Parent widget
        text (str): Button text
        command: Button command
        width (int): Button width
        height (int): Button height
        bg_color (str): Background color
        hover_color (str): Hover background color
        text_color (str): Text color
        font (tuple): Font configuration
        radius (int): Corner radius
        key (str): Unique key for this button
        state: Button state (tk.NORMAL, tk.DISABLED, etc.)
        
    Returns:
        tk.Button: Styled button
    """
    if not key:
        key = f"btn_{text}_{time.time()}"
    
    # Create normal and hover images
    normal_img = create_rounded_button(width, height, radius, bg_color, text, font, text_color, f"{key}_normal")
    hover_img = create_rounded_button(width, height, radius, hover_color, text, font, text_color, f"{key}_hover")
    
    # Create button with normal image and transparent background
    button = tk.Button(parent, image=normal_img, command=command, bd=0,
                       highlightthickness=0, bg="#f0f0f0", activebackground="#f0f0f0", state=state)
    
    # Store references to prevent garbage collection
    button.normal_img = normal_img
    button.hover_img = hover_img
    
    # Add hover effect
    def on_enter(e):
        e.widget.config(image=e.widget.hover_img)
    
    def on_leave(e):
        e.widget.config(image=e.widget.normal_img)
    
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    
    return button

def create_primary_button(parent, text, command, width=200, height=50, font=UI_FONTS["button"], radius=15, state=tk.NORMAL):
    """Create a primary styled button (green)"""
    return create_styled_button(
        parent, text, command, width, height, 
        UI_COLORS["primary"], UI_COLORS["hover_primary"], 
        UI_COLORS["button_text"], font, radius, state=state
    )

def create_secondary_button(parent, text, command, width=200, height=50, font=UI_FONTS["button"], radius=15):
    """Create a secondary styled button (orange)"""
    return create_styled_button(
        parent, text, command, width, height, 
        UI_COLORS["secondary"], UI_COLORS["hover_secondary"], 
        UI_COLORS["button_text"], font, radius
    )

def create_danger_button(parent, text, command, width=200, height=50, font=UI_FONTS["button"], radius=15):
    """Create a danger styled button (red)"""
    return create_styled_button(
        parent, text, command, width, height, 
        UI_COLORS["error"], "#D32F2F", 
        UI_COLORS["button_text"], font, radius
    )

def create_success_button(parent, text, command, width=200, height=50, font=UI_FONTS["button"], radius=15, state=tk.NORMAL):
    """Create a success styled button (green)"""
    return create_styled_button(
        parent, text, command, width, height, 
        UI_COLORS["success"], "#45A049", 
        UI_COLORS["button_text"], font, radius, state=state
    )

def create_styled_frame(parent, bg_color=UI_COLORS["background"], alpha=0.85, **kwargs):
    """
    Create a semi-transparent frame with rounded corners.
    
    Args:
        parent: Parent widget
        bg_color (str): Background color
        alpha (float): Transparency level (0-1)
        **kwargs: Additional frame options
        
    Returns:
        tk.Frame: Styled frame
    """
    frame = tk.Frame(parent, bg=bg_color, **kwargs)
    
    # Make semi-transparent if possible
    try:
        frame.attributes("-alpha", alpha)
    except:
        pass
    
    return frame

def setup_enhanced_background(window_or_frame, bg_image_path=None, window_key="default_bg"):
    """
    Set up an enhanced background with a background image.
    
    Args:
        window_or_frame: Window or frame to set background for
        bg_image_path (str): Path to background image
        window_key (str): Unique key for this background
        
    Returns:
        tk.Label: Background label
    """
    # Use provided path or default
    if not bg_image_path:
        bg_image_path = "background.jpg"
    
    # Try different paths if the image is not found
    image_paths = [
        bg_image_path,
        os.path.join(os.getcwd(), bg_image_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", bg_image_path),
        r"D:\Research\Research\Codes\Code_3\background.jpg"  # Fallback to absolute path
    ]
    
    original_image = None
    for path in image_paths:
        try:
            original_image = Image.open(path)
            print(f"Successfully loaded background image from: {path}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading image from {path}: {e}")
            continue
    
    if original_image is None:
        print("Could not find background image in any location. Using fallback color.")
        window_or_frame.configure(bg="lightblue")  # Fallback color
        return None
    
    try:
        
        # Get window/frame dimensions
        window_or_frame.update_idletasks()  # Ensure dimensions are calculated
        win_width = window_or_frame.winfo_width()
        win_height = window_or_frame.winfo_height()
        
        # If window is too small (unmapped), use default size
        if win_width < 100 or win_height < 100:
            win_width = 1920
            win_height = 1080
            try:
                if isinstance(window_or_frame, tk.Tk):
                    window_or_frame.geometry(f"{win_width}x{win_height}")
            except tk.TclError:
                pass
        
        # Resize image to window dimensions
        resized_image = original_image.resize((win_width, win_height), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(resized_image)
        
        # Create label for background with transparent background
        bg_label = Label(window_or_frame, image=bg_photo, bg="#f0f0f0")
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Keep reference to prevent garbage collection
        bg_label.image = bg_photo
        
        return bg_label
        
    except FileNotFoundError:
        print(f"Background image not found at: {bg_image_path}")
        window_or_frame.configure(bg="lightblue")  # Fallback color
    except Exception as e:
        print(f"Error setting background image: {e}")
        window_or_frame.configure(bg="lightgrey")  # Fallback color
    
    return None

def create_content_frame(parent, padding=20, bg_color="#f0f0f0", alpha=0.9):
    """
    Create a transparent content frame with padding.
    
    Args:
        parent: Parent widget
        padding (int): Padding around content
        bg_color (str): Background color (default: transparent)
        alpha (float): Transparency level (0-1)
        
    Returns:
        tk.Frame: Content frame
    """
    frame = tk.Frame(parent, bg=bg_color, padx=padding, pady=padding)
    
    # Remove shadow effect for transparent look
    frame.config(highlightthickness=0)
    
    return frame

def add_fade_animation(widget, duration=0.5, start_alpha=0.0, end_alpha=1.0):
    """
    Add fade-in animation to a widget.
    
    Args:
        widget: Widget to animate
        duration (float): Animation duration in seconds
        start_alpha (float): Starting alpha value
        end_alpha (float): Ending alpha value
    """
    steps = 10
    step_time = duration / steps
    alpha_step = (end_alpha - start_alpha) / steps
    
    # Hide widget initially
    widget.place_forget()
    
    def animate(step=0):
        if step <= steps:
            alpha = start_alpha + (alpha_step * step)
            try:
                widget.attributes("-alpha", alpha)
            except:
                pass
            widget.update()
            widget.after(int(step_time * 1000), lambda: animate(step + 1))
        
    # Show widget and start animation
    widget.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    animate()

def create_styled_label(parent, text, font=UI_FONTS["body"], fg=UI_COLORS["text"], bg="#f0f0f0", **kwargs):
    """Create a styled label with modern font and transparent background"""
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kwargs)

# ui_utils.py (add this function)

def create_title_label(parent, text, font=UI_FONTS["title"], fg=UI_COLORS["primary"], bg="#f0f0f0", **kwargs):
    """Create a styled title label with modern font"""
    # Use a regular Tkinter Label, as title labels typically don't need entry/combobox specific styling
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kwargs)

def create_styled_entry(parent, width=20, font=UI_FONTS["body"], **kwargs):
    """Create a styled entry field"""
    entry = tk.Entry(parent, width=width, font=font, **kwargs)
    entry.config(highlightbackground=UI_COLORS["primary"], highlightthickness=1)
    return entry

def create_styled_combobox(parent, values, width=20, font=UI_FONTS["body"], **kwargs):
    """Create a styled combobox"""
    combo = ttk.Combobox(parent, values=values, width=width, font=font, **kwargs)
    return combo

