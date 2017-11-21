# Hand recognition cnn
Hand-signs image recognition using 4 layers <b>convolutional neural network</b> using TensorFlow that recognize numbers from 0 to 5 in sign language

<ul>
<li>Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).</li>
<li>Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).</li>
</ul>

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/32612515-6c0dee0e-c5a3-11e7-82e7-1d872ffd022e.png" width="500"></p>

We implement a <b>4 layers convolutional neural network</b> using TensorFlow following next model: CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED. 
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33076989-6c4ed38c-cf09-11e7-87ac-fc4e5e604284.png" width="900"></p>

### hand_cnn_model.py
<ul>
<li>Train model</li>
<li>Save parameters in a checkpoint file in <b>/variables</b> folder</li>
</ul>

### cnn_utils.py
<ul>
<li>Contain supporting fuctions for hand_cnn_model.py and hand_cnn_reco.py</li>
</ul>

### hand_cnn_reco.py
<ul>
<li>Load trained model parameters from <b>/variables</b> folder checkpoint file </li>
<li>Lower chosen image resolution to 64 x 64 pixels</li>
<li>Run chosen image into the network providing prediction </li>
<li>Example:</li>
</ul>



<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33071760-cf2a1a50-cef7-11e7-8969-f3700c428dd4.png" width="500"></p>
