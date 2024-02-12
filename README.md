# Mobile Fitness App

A simple mobile fitness app that counts the number of repetitions of fitness exercises using computer vision & machine learning. [Live Demo](https://joe9154.github.io/Mobile-Fitness-App/)


## How does it work?

- The app captures camera feed and sends it through [Movenet](https://www.tensorflow.org/hub/tutorials/movenet) pose detection model. The model returns 17 body keypoints.
- Detected keypoints are sent through a feed-forward neural network. The output are probabilities of the upper and lower pose of an exercise (e.g. deadlift_up, deadlift_down).
- An algorithm uses this data to count the number of repetitions.

![diagram](/readme_resources/diagram_eng.png)

## Built with

- HTML, JS, CSS, [Bootstrap](https://getbootstrap.com/)
- [Tensorflow.js](https://www.tensorflow.org/js)
- [Movenet](https://www.tensorflow.org/hub/tutorials/movenet)

## How to install?

The app uses CDNs for all it's dependencies, so no need to install anything. Just download it and serve it.

## How to use it?

- On the homepage select the exercise you wish to perform.
- Set your camera to a location where your whole body is visible.
- Click "Start". The app will start loading. Then a countdown timer will appear. When it reaches 0, you can start performing the exercise.
- The app will automatically count the number of repetitions you did.

It is important to note that the app is by no means perfect. The accuracy of the number of recognized repetitions can vary widely by:
- the location and angle of the camera
- the lighting in the room

This is because the training data set for the neural net was quite small. With a bigger data set and more dynamic pose examples in it, a bigger accuracy could be achieved.

