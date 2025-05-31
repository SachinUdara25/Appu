# sensor_function.py

import serial
import serial.tools.list_ports
import time
from tkinter import messagebox
import utils # Ensuring we can access utils.arduino

def find_arduino_port():
    """Automatically finds the port where an Arduino is connected."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Common descriptions for Arduino devices
        if "Arduino" in port.description or            "USB Serial" in port.description or            "CH340" in port.description or            "CP210x" in port.description: # Added another common chip
            return port.device
    return None

def connect_arduino():
    """Connects to the Arduino on the automatically found port."""
    port = find_arduino_port()
    if port:
        try:
            utils.arduino = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # Wait for the connection to establish
            print(f"Connected to Arduino on {port}")
            messagebox.showinfo("Arduino Status", f"Connected to Arduino on {port}")
        except serial.SerialException as e:
            messagebox.showerror("Arduino Error", f"Could not connect to Arduino on {port}: {e}")
            utils.arduino = None
        except Exception as e: # Catch any other unexpected errors
            messagebox.showerror("Arduino Error", f"An unexpected error occurred while connecting to Arduino: {e}")
            utils.arduino = None
    else:
        messagebox.showwarning("No Arduino Found", "Arduino not found. Please ensure it's connected.")
        utils.arduino = None

def get_distance():
    """Requests and retrieves a distance reading from the Arduino."""
    if utils.arduino is None or not utils.arduino.is_open:
        # messagebox.showerror("Arduino Error", "Arduino not connected. Cannot get distance.")
        print("Arduino not connected. Cannot get distance.")
        # Attempt to reconnect if not connected
        # connect_arduino()
        # if utils.arduino is None or not utils.arduino.is_open:
        return -1 # Return -1 or some indicator of error/no reading

    try:
        utils.arduino.flushInput() # Clear input buffer before sending command
        utils.arduino.write(b'D')  # Send command to Arduino to request distance
        line = utils.arduino.readline().decode('utf-8', errors='ignore').strip()
        if line:
            dist = float(line)
            # Assuming valid distances are between a certain range, e.g., 50mm to 1000mm
            # The original code had 50 < dist < 1000. Keeping this logic.
            if 50 < dist < 1000:
                return dist
            else:
                # print(f"Distance out of expected range: {dist}")
                return -2 # Or some other error code for out-of-range
        else:
            # print("No data received from Arduino for distance.")
            return -3 # Or some other error code for no data
    except serial.SerialTimeoutException:
        # print("Timeout reading from Arduino.")
        return -4 # Error code for timeout
    except ValueError: # Handles cases where float conversion fails
        # print(f"Invalid data received from Arduino: {line}")
        return -5 # Error code for invalid data format
    except Exception as e:
        # print(f"Error reading distance from Arduino: {e}")
        # messagebox.showerror("Arduino Communication Error", f"Error reading distance: {e}")
        return -6 # Generic error code

if __name__ == '__main__':
    # Example usage for testing sensor_function.py directly
    print("Attempting to connect to Arduino...")
    connect_arduino()
    if utils.arduino:
        print("Arduino connected. Attempting to get distance measurements...")
        for i in range(10):
            distance = get_distance()
            if distance > 0:
                print(f"Distance: {distance} mm")
            elif distance == -1:
                print("Failed to get distance: Arduino not connected.")
                break
            elif distance == -2:
                print("Failed to get distance: Value out of expected range.")
            elif distance == -3:
                print("Failed to get distance: No data received.")
            elif distance == -4:
                print("Failed to get distance: Read timeout.")
            elif distance == -5:
                print("Failed to get distance: Invalid data format from Arduino.")
            elif distance == -6:
                print("Failed to get distance: Generic error.")
            time.sleep(1)
        utils.arduino.close()
        print("Arduino connection closed.")
    else:
        print("Failed to connect to Arduino.")
