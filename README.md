# Real-time Facial Emotion Recognition Web Application

This repository hosts a cutting-edge web application designed for real-time facial emotion recognition, blending Flask, Python, HTML, CSS, and advanced machine learning models. It's distinguished by its use of three pivotal models: Convolutional Neural Network (CNN), Chehra model, and Deep Neural Network (DNN), each meticulously trained for high accuracy in detecting diverse emotional states. A hallmark of this project is its interactive visualization, which dynamically showcases emotion detection results and graphically represents them for insightful analysis.

## Project Overview

Understanding and analyzing human emotions in real time can significantly impact various domains such as mental health assessment, user experience enhancement, and educational tools. Our web application offers a robust and user-friendly platform for real-time analysis and categorization of human emotions through facial expressions, leveraging cutting-edge AI and human-computer interaction advancements.

## Key Features

- **Real-time Emotion Detection:** Detect and analyze facial emotions in live video streams using your web camera.
- **Interactive Visualization:** Dynamic display of detected emotions on the web page, along with a bar chart graph for an in-depth analysis over time.
- **Model Flexibility:** Switch seamlessly between CNN, Chehra, and DNN models, catering to various needs for accuracy and computational efficiency.
- **User-Friendly Interface:** An intuitive and navigable front-end ensures a smooth user experience.
- **Comprehensive Reporting:** Generate detailed reports of detected emotions for further analysis.

## Motivation

The project aims to bridge the gap between human emotions and technology, providing insights into emotional dynamics and enhancing interactions in digital spaces. It stands as a testament to the potential applications of machine learning in real-time emotion detection and interactive visualization.

Here's the revised Installation and Setup section with an added step for setting up a virtual environment:

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gmMustafa/facial_emotion_detection/
   ```

2. **Download the architecture and model zip file** from the following URL, and extract it into the root folder of the cloned repository:
   ```plaintext
   https://tinyurl.com/439nt2th
   ```

3. **Set up a Python virtual environment:**
   Navigate to your project directory and create a virtual environment. Activate it before proceeding to the next steps.
   - **Create the virtual environment:**
     ```bash
     python -m venv venv
     ```
   - **Activate the virtual environment:**
     - On **Windows**, run:
       ```cmd
       venv\Scripts\activate
       ```
     - On **macOS/Linux**, run:
       ```bash
       source venv/bin/activate
       ```

4. **Install dependencies:**
   With the virtual environment activated, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch the application:**
   Ensure the virtual environment is still activated, then start the Flask application:
   ```bash
   python app.py
   ```

6. **Access the application** by navigating to `http://127.0.0.1:5000/` in your web browser.

Remember to deactivate your virtual environment when you're finished working on the project by running `deactivate` in your terminal.
   
## Usage Guide

- Grant the application permission to use your web camera.
- Select the desired emotion detection model from the dropdown menu.
- Experience real-time emotion detection and visualization.
- Utilize the "Generate Report" feature for a session summary of detected emotions.

## Technologies Used

- **Flask:** Serves as the backbone for server-side operations.
- **HTML/CSS/JavaScript:** Crafts a responsive and interactive UI.
- **TensorFlow/Keras:** For model training and inference.
- **OpenCV:** For video stream processing and face detection.
- **Dlib:** Facial landmark detection in the Chehra model.



## Contributors

A huge thank you to everyone who has contributed to this project! Your contributions help make this project better.

- [@Ali623](https://github.com/Ali623) - **Aliullah**
- [@FidaHussain87](https://github.com/FidaHussain87) - **Fida Hussain**
- [@hamzanaeem1999](https://github.com/hamzanaeem1999) - **Hamza Naeem**


## Screenshots

### UI Overview
![UI Overview](https://github.com/gmMustafa/facial_emotion_detection/assets/26876754/cfaa212d-6665-4ca7-a932-bebb2139eecf)

![UI Overview - II](https://github.com/gmMustafa/facial_emotion_detection/assets/26876754/5f601fc8-7b71-4d75-af79-d81bdccb75d9)


## Contributing

We welcome contributions to improve functionality, model performance, or user experience. Please fork the repository and submit pull requests for review.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE.md) for details.

## Acknowledgments

Our heartfelt thanks go to the facial emotion recognition community for their research and datasets that have significantly contributed to this project's success.

## Future Directions

Future enhancements will focus on integrating more advanced neural networks, refining user interfaces, and expanding the application's reach into various sectors, guided by ethical considerations and a commitment to innovation.

## Disclaimer
This project is associated with the "Interactive Visualization Project" under the "AI Systems and Applications" pillar, part of the Master of Science in Artificial Intelligence degree at Friedrich-Alexander-Universität Erlangen-Nürnberg.

For a detailed exploration of the project's methodology, evaluation, and insights, refer to the presentation slides at https://tinyurl.com/4ppxptxp.

<img src="https://github.com/gmMustafa/facial_emotion_detection/assets/26876754/d94d54c8-a95d-4bd0-935a-6a714e6bb909" alt="Friedrich-Alexander-Universität Erlangen-Nürnberg" width="250" height="80">


---
