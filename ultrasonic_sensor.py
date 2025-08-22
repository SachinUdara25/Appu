#ultrasonic_sensor.py
"""
Reads distance from an ultrasonic sensor via an Arduino connected over a serial port.

This script attempts to automatically detect the serial port to which an
Arduino-like device is connected. Once connected, it continuously reads
distance data sent by the Arduino.

The Arduino is expected to:
1. Read distance from an ultrasonic sensor (e.g., HC-SR04).
2. Send the measured distance as a string over the serial connection.
3. The string should represent the distance in MILLIMETERS.
4. The string should be newline-terminated (e.g., by using `Serial.println(distance_in_mm)`
   in the Arduino sketch).

The script then prints the received distance to the console in the format:
"Distance - xx.xx mm"
"""

import serial
import serial.tools.list_ports
import time
import re

# ------------------------------------------------------------------------------------
# ARDUINO SETUP REQUIREMENTS:
#
# To use this Python script, your Arduino should be programmed to:
# 1. Interface with an ultrasonic sensor (e.g., HC-SR04, JSN-SR04T).
# 2. Measure the distance.
# 3. Send the measured distance as a numerical string over the USB serial connection.
# 4. The distance sent should be in MILLIMETERS.
# 5. Each distance reading sent over serial MUST be terminated with a newline character
#    (e.g., by using `Serial.println(distance_in_mm);` in your Arduino code).
# 6. A baud rate of 9600 is recommended to match this script's default.
#
# Example Arduino `loop()` snippet:
# ```cpp
#   long duration;
#   int distance_mm;
#   // Assuming 'trigPin' and 'echoPin' are defined for your sensor
#   digitalWrite(trigPin, LOW);
#   delayMicroseconds(2);
#   digitalWrite(trigPin, HIGH);
#   delayMicroseconds(10);
#   digitalWrite(trigPin, LOW);
#   duration = pulseIn(echoPin, HIGH);
#   distance_mm = duration * 0.34 / 2; // Or other appropriate conversion for your sensor
#   Serial.println(distance_mm);
#   delay(100); // Send reading approximately every 100ms
# ```
# ------------------------------------------------------------------------------------

class UltrasonicSensor:
    """
    A class to interface with an ultrasonic sensor that sends data via
    an Arduino (or similar microcontroller) over a serial connection.

    The class handles auto-detection of the serial port, connection,
    and reading/parsing of distance data.
    """

    def __init__(self, port=None, baud_rate=9600, timeout=1):
        """
        Initializes the ultrasonic sensor interface.

        Args:
            port (str, optional): The serial port the sensor (Arduino) is connected to.
                                  If None, it attempts to find the sensor automatically
                                  by looking for common Arduino-related keywords in port
                                  descriptions or manufacturers. Defaults to None.
            baud_rate (int, optional): The baud rate for serial communication.
                                       Must match the Arduino's baud rate. Defaults to 9600.
            timeout (int, optional): The serial communication timeout in seconds
                                     for read operations. Defaults to 1.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None  # Will hold the serial.Serial object

        if self.port is None:
            # Attempt to find the port if none is specified.
            self._auto_detect_port()

        if self.port is not None:
            try:
                # Establish the serial connection.
                self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
                print(f"Connected to ultrasonic sensor on port {self.port} at {self.baud_rate} baud.")
                # Allow some time for the serial connection to establish and for the
                # Arduino to potentially reset after connection.
                time.sleep(2)
            except serial.SerialException as e:
                print(f"Error: Could not open serial port {self.port}: {e}")
                self.ser = None  # Ensure ser is None if connection failed
        else:
            # This message is printed if auto-detection failed and no port was specified.
            print("Error: No serial port specified and no compatible sensor auto-detected.")

    def _auto_detect_port(self):
        """
        Attempts to automatically detect the serial port of an Arduino-like device.

        It iterates through available serial ports and checks their descriptions
        and manufacturer information for common keywords associated with Arduinos
        or USB-to-serial converters (e.g., "arduino", "ch340", "usb serial").
        The first port matching any keyword is selected.
        Sets `self.port` to the detected device path if found.
        """
        print("Attempting to auto-detect serial port for a compatible sensor (e.g., Arduino based)...")
        ports = serial.tools.list_ports.comports()
        keywords = ["arduino", "ch340", "usb serial"]  # Common keywords for Arduino/serial adapters

        for p in ports:
            port_description = p.description.lower() if p.description else ""
            port_manufacturer = p.manufacturer.lower() if p.manufacturer else ""

            print(f"Checking port: {p.device} - Description: '{p.description}' - Manufacturer: '{p.manufacturer}'")

            for keyword in keywords:
                if keyword in port_description or keyword in port_manufacturer:
                    print(f"Compatible device keyword '{keyword}' found for port {p.device}.")
                    print(f"Setting {p.device} as the target port.")
                    self.port = p.device  # Set the found port
                    return  # Exit after finding the first potential port

        if not self.port:
            # This message is printed if the loop completes without finding a suitable port.
            print("Auto-detection failed: No compatible sensor port (e.g. Arduino, CH340, USB Serial) found.")

    def get_distance(self):
        """
        Reads a distance measurement from the serial port.
        Returns the distance in millimeters (float) or None if reading fails.
        """
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port not open. Cannot get distance.")
            return None

        try:
            # Read a line from the serial port
            line = self.ser.readline()
            if not line:
                # No data received within the timeout period
                # print("No data received from sensor (timeout).")
                return None

            try:
                # Decode using latin-1 to handle problematic bytes like 0xfc
                decoded_line = line.decode('latin-1').strip()
            except UnicodeDecodeError as e:
                print(f"Decoding error (latin-1 fallback failed): {e} - Raw bytes: {line}")
                return None
            except Exception as e:
                print(f"Unexpected decoding error: {e} - Raw bytes: {line}")
                return None

            # Use regex to extract numeric data
            match = re.search(r'(\d+\.?\d*)', decoded_line)
            if match:
                distance_str = match.group(1)
                try:
                    distance = float(distance_str)
                    # Validate the range (e.g., 10mm to 4000mm) for realistic values
                    # (Adjust min/max based on your sensor's actual range)
                    if 0.1 <= distance <= 5000.0: # Ensure distance is within a plausible range
                        return distance
                    else:
                        print(f"Invalid distance reading: {distance} (outside range 0.1-5000mm) from '{decoded_line}'")
                        return None
                except ValueError:
                    print(f"Could not convert '{distance_str}' to float. Decoded: '{decoded_line}'")
                    return None
            else:
                # No numeric data found in the decoded line
                # print(f"No numeric data found in sensor reading: '{decoded_line}'")
                return None

        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
            self.close() # Close port on error
            return None
        except Exception as e:
            print(f"An unexpected error occurred during sensor reading: {e}")
            return None

    def get_distance_with_retry(self, max_retries=3):
        """
        Reads distance with retry mechanism for handling invalid readings.
        Returns the distance in mm as a float, or None if all retries fail.
        """
        for attempt in range(max_retries):
            distance = self.get_distance()
            if distance is not None:
                return distance
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                time.sleep(0.1)  # Small delay between retries
        
        print(f"Failed to get valid distance reading after {max_retries} attempts")
        return None

    def read_distance(self):
        """
        Alias for get_distance_with_retry() method to provide better reliability.
        Uses retry mechanism by default for better error handling.
        
        Returns:
            float: The distance in millimeters, or None if an error occurs.
        """
        return self.get_distance_with_retry()

    def close(self):
        """
        Closes the serial connection if it is open with timeout handling.
        """
        if self.ser and self.ser.is_open:
            try:
                # Set a timeout for cleanup operations to avoid indefinite waits
                original_timeout = self.ser.timeout
                self.ser.timeout = 0.5  # 0.5 second timeout for cleanup
                
                # Try to flush any remaining data
                try:
                    self.ser.flushInput()
                    self.ser.flushOutput()
                except:
                    # Ignore flush errors during cleanup
                    pass
                
                # Restore original timeout before closing
                self.ser.timeout = original_timeout
                
                self.ser.close()
                print(f"Serial port {self.port} closed.")
                
            except serial.SerialException as e:
                # This might happen if the device is abruptly disconnected
                print(f"Error closing serial port {self.port}: {e}")
            except Exception as e:
                print(f"Unexpected error during port closure: {e}")
            finally:
                # Ensure ser is set to None regardless of close success
                self.ser = None
        else:
            # If called when port was never opened or already closed.
            print("Serial port was not open or not initialized, no action taken to close.")

if __name__ == '__main__':
    print("Starting Ultrasonic Sensor (Serial) example program.")

    # Create an instance of the UltrasonicSensor.
    # This will attempt to auto-detect and connect to the Arduino.
    sensor = UltrasonicSensor()

    # Check if the sensor's serial port was successfully initialized.
    if sensor.ser and sensor.ser.is_open:
        print("Sensor connection successful. Starting distance readings...")
        try:
            # Loop indefinitely to get continuous readings.
            while True:
                # Retrieve distance from the sensor with retry mechanism
                distance = sensor.get_distance_with_retry()

                if distance is not None:
                    # Print the successfully read distance, formatted to two decimal places.
                    # The distance is expected to be in millimeters as per sensor setup.
                    print(f"Distance - {distance:.2f} mm")
                # If distance is None, the get_distance_with_retry() method already prints
                # specific error or status messages (e.g., timeout, parse error).

                # Wait for a short period before the next reading.
                time.sleep(0.1)  # Read approximately 10 times per second.

        except KeyboardInterrupt:
            # This block is executed if the user presses Ctrl+C.
            print("Exiting program due to Ctrl+C...")
        finally:
            # This block is always executed, whether the loop exits normally
            # (which it won't in this while True structure without a break)
            # or due to an exception (like KeyboardInterrupt).
            print("Closing sensor connection...")
            sensor.close()  # Ensure the serial port is closed.
    else:
        # This block is executed if sensor.ser is None or the port is not open,
        # meaning initialization in sensor.__init__() failed.
        # Specific error messages should have already been printed by the class methods.
        print("Failed to initialize or connect to the sensor. Please check setup and permissions. Exiting program.")

    print("Ultrasonic Sensor (Serial) example program finished.")

# End of script


