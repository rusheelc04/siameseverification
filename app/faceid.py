"""
Kivy application that uses a Siamese neural network (via TensorFlow) to verify user identity 
in real-time using a webcam feed.

Steps of operation:
1. Capture a frame from the webcam.
2. Crop, resize, and normalize the frame for model input.
3. Compare the captured frame with stored verification images using a Siamese model.
4. Determine if the user is verified based on specified detection and verification thresholds.
"""

# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

class CamApp(App):
    """
    Main Kivy Application Class for Face Verification.
    Uses a pre-trained Siamese neural network model to verify user identity.
    """

    def build(self):
        """
        Initialize and build the GUI layout:
        1. Create the Image widget for the webcam feed.
        2. Create the 'Verify' Button widget.
        3. Create a Label to display verification status.
        4. Load the Siamese model.
        5. Start capturing the webcam feed.
        """
        # Main layout components
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(
            text="Verify",
            on_press=self.verify,
            size_hint=(1, 0.1)
        )
        self.verification_label = Label(
            text="Verification Uninitiated",
            size_hint=(1, 0.1)
        )

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load TensorFlow/Keras model
        self.model = tf.keras.models.load_model(
            'siamesemodel.h5',
            custom_objects={'L1Dist': L1Dist}
        )

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)

        # Schedule the update function to run 33 times per second
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        """
        Continuously capture frames from the webcam,
        crop them to the region of interest, flip vertically,
        and update the Kivy Image widget's texture.
        """
        # Read frame from OpenCV
        ret, frame = self.capture.read()
        # Crop the region of interest
        frame = frame[120:120+250, 200:200+250, :]

        # Flip frame vertically (for a mirrored webcam effect) and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr'
        )
        img_texture.blit_buffer(
            buf,
            colorfmt='bgr',
            bufferfmt='ubyte'
        )
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        """
        Load an image from file_path, resize it to 100x100,
        normalize pixel values to [0,1], and return the processed tensor.
        """
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Decode the image
        img = tf.io.decode_jpeg(byte_img)

        # Resize image to 100x100
        img = tf.image.resize(img, (100, 100))
        # Normalize pixel values
        img = img / 255.0

        return img

    def verify(self, *args):
        """
        Verification function to capture the current webcam frame,
        compare it against the stored verification images, and determine
        if the user is verified based on detection and verification thresholds.
        """
        # Specify thresholds
        detection_threshold = 0.99
        verification_threshold = 0.8

        # Capture input image from our webcam
        SAVE_PATH = os.path.join(
            'application_data',
            'input_image',
            'input_image.jpg'
        )
        ret, frame = self.capture.read()
        # Crop the region of interest
        frame = frame[120:120+250, 200:200+250, :]
        # Save the captured frame
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        for image in os.listdir(
            os.path.join('application_data', 'verification_images')
        ):
            # Preprocess the input image
            input_img = self.preprocess(
                os.path.join('application_data', 'input_image', 'input_image.jpg')
            )
            # Preprocess the validation image
            validation_img = self.preprocess(
                os.path.join('application_data', 'verification_images', image)
            )

            # Make prediction with the Siamese model
            result = self.model.predict(
                list(np.expand_dims([input_img, validation_img], axis=1))
            )
            results.append(result)

        # Detection threshold: metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification threshold: proportion of positive predictions over total samples
        verification = detection / len(
            os.listdir(os.path.join('application_data', 'verification_images'))
        )
        verified = verification > verification_threshold

        # Update label text based on verification result
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details to the console
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified

# Entry point of the application
if __name__ == '__main__':
    CamApp().run()