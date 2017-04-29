Detection Images (All were run with heat map threshold of 2 and a sliding frame of 8 pixels (Note, different sampling rate was used in video model due to speed and the improved accuracy from buffering image history)

detection1: (Correct) Two clear car detections, black car appearing more prominent than the white likely due to contrast and perspective

detection2: (Correct) No cars detected in open road

detection3: (Incorrect) No cars detected, white car off in the distance but should have been detected

detection4: (Correct) Same reasoning and conclusion as image 1

detection5: (Correct/Incorrect): Picked up the black car but missed the white one. May be due to trimming the x-dimension.

detection6: (Correct) Same reasoning and conclusion as image 1



Histogram of Oriented Gradients: Not the stark difference int he histogram image outputs, indicating something of a signature for the automobile vs.tree 

hog1: Automobile image in YUV color space with each color channel HOG.
hog2: Non-automobile image in YUV color space with each color channel HOG.