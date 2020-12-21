# ASL Fingerspelling Prediction
This project consists of an application which can recognize and display correct alphabet sign classes from the captured video of a person signaling ASL Alphabet signs.

The project consists of two major steps.

The first step involves successfully identifying the position of both hands in the video, processing it and cropping out only hand palms. The open source solution used in this step is from https://github.com/victordibia/handtracking.

Once the hand palm is identified and cropped, the second step involves applying CNN model to automatically infer the type of the ASL Alphabet signs being displayed. The model is trained on ASL Alphabet signs dataset at https://www.kaggle.com/grassknoted/asl-alphabet.

### Tasks:
1.	Use mobile camera / webcam to collect data on all 26 alphabets [frame should be entire upper body].
2.	Develop a palm detection algorithm.
a.	Use Posenet to obtain wrist points.
b.	Develop a cropping algorithm from empirical understanding of the video data.
c.	Validate the palm detection algorithm on the videos collected.
3.	Configure a 2D CNN that is trained on the ASL alphabet signs data set in the given link.
4.	Use own videos to recognize which alphabets were signed [80% re-training, 20% validation].
5.	Report accuracy F1 score metrics.
6.	Now take 40 different words and use the alphabet signs to fingerspell these words.
a.	Shoot videos of each of these words using interface used in Task 1.
7.	Use Posenet on each of these videos to develop a keypoint time series.
8.	Develop a segmentation algorithm that separates each alphabet in the word.
9.	Use the same 2D CNN to recognize each alphabet in the word.
10.	Develop an algorithm that combines each alphabet recognition result to develop the recognition of a word.
11.	Report the word recognition accuracy.

#### Steps to Run:
Pre-requisite libraries: Install TensorFlow and OpenCV in your system.
1. Clone the repository
`git clone https://github.com/debarshi25/ASL-Fingerspelling-Prediction.git`
2. Record ASL Fingerspelling videos in .mp4 format and save in "alphabet_videos/demo" and "word_videos/demo" for alphabets and words respectively.
3. Run the application
`python3 prediction.py`