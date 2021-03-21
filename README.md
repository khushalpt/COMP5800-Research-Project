Data dimensionality is the total number of variables being measured in every observation. With increasing trends in technology, there is a huge volume of data that is being created. One such field of technology is computer vision. Human beings are able to detect and recognize faces with ease even with external conditions such as expressions, illuminations or viewing angle affecting the sight when compared to the machines. This is because of high dimensions associated with it. The way forward is by reducing the dimensions that in turn helps in minimizing the with-in class distances. This project aims to compare different applicable dimensional reduction techniques suitable for facial recognition system and propose an ensemble model of such techniques that will help improve the accuracy of the model and gauge the performance by testing it with different datasets consisting of facial images with varying illuminations, complex backgrounds, and expressions. The proposed ensemble model extracts feature vectors using a hybrid of two dimensional reduction techniques – principal component analysis and locally linear embedding, and pass them through dense convolutional neural network to predict faces. The model performs with a testing accuracy of 0.95 and a testing F1 score of 0.94. on labelled faces in the wild dataset. 

**Test Cases**
Test case 1
![testcase1](https://user-images.githubusercontent.com/16033184/111923913-05073880-8a78-11eb-8cac-baa1f917519a.png)

Test case 2 -- Occlusion
![testcase2_occlusion](https://user-images.githubusercontent.com/16033184/111923919-0fc1cd80-8a78-11eb-8f6e-1589044be440.png)

Test case 3 -- illumination
![testcase3-illumination](https://user-images.githubusercontent.com/16033184/111923927-194b3580-8a78-11eb-9111-b166a20e59f8.png)

Test case 4 -- orientation
![testcase4-orientation](https://user-images.githubusercontent.com/16033184/111923930-21a37080-8a78-11eb-80e4-5059e966e723.png)

Test case 5 -- image uploader
![testcase5-image uploader](https://user-images.githubusercontent.com/16033184/111923939-2b2cd880-8a78-11eb-913c-45b25960f867.png)

Test case 6 -- personality recognition using phone
![testcase6-famous personality](https://user-images.githubusercontent.com/16033184/111923968-54e5ff80-8a78-11eb-814e-278caefb6ac3.png)


