The goals / steps of this project are the following:
DONE - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
DONE - Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
DONE - Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
DONE - Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
DONE - Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
DONE - Estimate a bounding box for vehicles detected.




--- Histogram of Oriented Gradients (HOG) ---
1. Explain how (and identify where in your code) you extracted HOG features from the training images.
2. Explain how you settled on your final choice of HOG parameters.
3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


(Refer to lines approx. 110-130 in core.py for extraction implementation. The function is used in the extract_features() and pipeline() functions.)

The Histogram of Orientation Gradients (HOG) features were extracted from each of the training images using Scikit-Image Feature Processing library to extract a cell-blocked histogram of gradients. The purpose of this library is to construct an abstracted view of each pixels color gradient orientation within an image, greatly increasing the significance of image characteristics (e.g., edges, changes of color, etc.) while reducing the number of features needed for analysis/classification. Implementation was straight forward, whether extracting the vectorized features for each color channel or 'blocked' features for subsample (more on that to come) was straight forward, due to the function provided by the library. 

All training images were 64x64 pixels and based upon the lessons instruction it was natural to start with the parameters provided. The image is broken into 64 8x8-cells, composing a total of 49 blocks (incremented by one cell per extraction (64 pixels minus the original 8 pixel cell sample yields a total of 7 samples per X or Y dimension)), each of which are analyzed for each pixel gradient and grouping into their directional bins (orientations). Therefore, a total feature set of 1764 is produced per channel of the training image (as opposed to 4096 pixel gradients per channel). 

I read the article provided in the lesson and found they preferred a 6x6 pixel cell dimension with 3x3 cell blocks; however, I found the initial parameters to do better during testing and maintained those through the rest of my project.


(NOTE: (refer to lines appox. 75-105) I augmented the HOG features with spatial binning (pxl_extract --- resize the image the a smaller scale, effectively averaging neighboring pixels) and color channel histogram (chnl_extract --- analyze each color channel and bin the color values into predefined ranges) features. Both techniques are a way to reduce an image into a broader characterization that can provide value when there is sharp color/saturation/exposure contact between an object (e.g., an automobile) and the surroudings.)


------------------------------------------------------------------------------------------------------------------------------------------------------------------------


--- Sliding Window Search ---
1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
--- Video Implementation ---
3. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
4. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.



(NOTE: All images can be found in the [pipleline_output] directory in the repository with a README_IMAGES describing each. Video is also provided.)

(Refer to lines approx. 210-290 in core.py for the sliding window technique. NOTE: An option is provided to use HOG subsampling; however, I found that detection did not perform as well as expected. Asking other students, it appears that there is nothing incorrect about how I implemented the extraction and since there were no signifcant perfomance implications in image/video extraction (approx. 60-80% difference in speed) I decided to perform each analysis one sample at a time.)


The sliding window search can be found in the pipeline() function and broken into four main sections: 

1) Defining the image scales to search within a trimmed image (approx. lines 220-235)
2) Trim and scale image, extracting the HOG features for the entire image (approx. lines 235-255)
3) Perform the X/Y sliding image sample search, deriving the features for the sample, then passing them through a classifier and adding them if an automobile is detected (approx. lines 255-285)
4) Add found frames to the buffer, generate the heatmap, and return the most up to date automobile frames (approx. lines 285-290, 305-365)


The scales that were chosen were based upon reviewing the test images that were provided. The automobiles in the foreground would clearly appear larger than those that are far away, making more than one scale necessary to keep track of them as they changed distance from the camera. I chose two scaled (1.25 and 1.5) to detect those that further away (still larger than a 64x64 pixel training image) and those that are closer, respectively. Trying scales much larger than 1.5 did not drastically improve detection and yielded fewer searching frames that led to missed detections.

Once I had determined adequate scales (note that all of the alterations I made to the pipeline were done in an iterative fashion) where true positives appeared to be consistent, I needed a way to remove false positive that appeared occasionally in frames. Knowing that the false negative detections would appear in one frame and not the next made it a prime candidate to maintain a history (i.e., buffer) of previous detections to separate true postives from false. With the gathered frames (in my case I stored the heatmaps instead of the individual frames), I could compress all of the generated heatmaps they generated into a single view of the current image. I then applied a threshold (tested a range from 2-8) to remove all pixels that didnt accumulate more than the required number of detections within a given frame. These heatmaps could then be labeled, essentially defining distinct bounds around the "hottest" points, and return defining frames to the pipeline for drawing on top of the image.

Throughout the slinding window search exercise (not including the feature training parameter decisions) I adjusted: image trimming (X/Y search area bounds), heatmap threshold, buffer size, HOG subsampling techniques, decision_function bounds, and image scaling.






------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Stupid mistakes:
- Sobel thresholds: Attempted to apply a Sobel magnitude operator in the HOG extraction after reading about it in a paper, thinking that it would be the gradients even more distinct. I believe this greatly reduced what an automobiles gradient characteristics are to that which any non-car image could have.
- Sliding window over half the image: Initial implementations searched to large of an area for vehicles, making the pipepline testing and validation very slow.
- Trying to many dense functions: Simpler is better. Dense, overloaded functions are not always better. It lead to a lot of complicated debugging and I think due to the lack of time I was trying to pack it all in to quickly...

Current faults / Improvements to make:
- False positives: Detecting only true positives is not perfect at this point. There are a number of ways I can tune the color spaces, features extractions/training, and heatmap generation to improve this exercise. In the video provided there are a number of instances where false positives still arise, mainly coming from the on-coming traffic (Note, I read some suggestions of just trimming the left hand side search area even more, but that seems like a lazy approach. If the camera perspective changes to the far right hand side of the lane, I should not trim that area....obviously a more dynamic detection approach would be required for a fully implemented approach.)
- HOG subsampling: Checking with over the subsampling code very heavily, and asking other students, I could not find anything incorrect about my implementation but my results for the subsampling technique always appeared worse than if I extracted features from each individual sample. I do not have an explanation for the difference but did test it extensively.
- Frame confidence: It would be great to use the decision_function to partition the sample preditions into different buffers and weight them in the heatmap, therefore it would no longer be a "CAR/NO-CAR" scenario but a "I think that is a side of a car, a backend, etc....but that is definitely a tree".
- Higher sampling rates: Change the number of samples taken from each step of the video frame to increase the number of positive detections found (also increasing the heatmap threshold to remove increased amount of false positives). The sampling rate chose (32) was used due to the balance between speed and accuracy.



