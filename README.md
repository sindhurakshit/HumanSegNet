# HumanSegNet
HumanSegNet
# Person Image Segmentation with PyTorch
Image segmentation is a computer vision technique that involves dividing an image into different regions or segments based on certain characteristics or criteria. The goal of image segmentation is to partition an image into meaningful and semantically coherent regions to facilitate further analysis or understanding of the image.

Image segmentation has numerous applications in various fields, including medical imaging, autonomous driving, object recognition, image editing, and augmented reality. It plays a crucial role in tasks like object detection, image understanding, and scene analysis by providing a more detailed and structured representation of the visual content in an image.

There are various approaches available for segmentation like - 
<ul>
<li><b>Thresholding: </b>Assigning pixels to different segments based on a threshold value of intensity or color.
Region-based segmentation: Grouping pixels with similar properties, such as color, texture, or intensity, to form coherent regions.
Edge detection: Identifying boundaries or edges between different objects or regions in an image by analyzing changes in intensity or color gradients.
</li> 
  
<li><b>Clustering:</b> Using algorithms like K-means or Gaussian Mixture Models (GMM) to group pixels into clusters based on feature vectors.
Deep learning-based segmentation: Utilizing Convolutional Neural Networks (CNNs) and other deep learning architectures to learn complex representations and accurately segment images.
</li>
<li><b>Watershed segmentation: </b>Treating an image as a topographic map and flooding regions from markers to separate objects or regions.
Graph-based segmentation: Representing an image as a graph and dividing it into segments based on graph properties, such as edge weights or connectivity.
  </li>
</li>
<li><b>Active contour models (Snakes):</b> Deformable models that iteratively adjust their shape to capture object boundaries based on energy minimization.
  </li>
<li><b>Superpixel segmentation: </b>Dividing an image into perceptually meaningful and compact regions, known as superpixels, to reduce complexity and facilitate subsequent analysis.
  </li>
<li><b>Multi-scale segmentation: </b>Performing segmentation at multiple scales or resolutions to capture both fine and coarse details.
<li><b>Hybrid approaches: </b>Combining multiple segmentation techniques or incorporating domain-specific knowledge to improve segmentation accuracy and robustness.
</li>
</ul>

<b>Deep learning-based segmentation:</b> Convolutional Neural Networks (CNNs) and other deep learning architectures have shown significant advancements in image segmentation. Models like U-Net, Mask R-CNN, and FCN (Fully Convolutional Network) utilize neural networks to learn complex representations and accurately segment images.

In this project, we will use human detection Human Segmentation Dataset originally created by VikramShenoy97 and available at https://github.com/VikramShenoy97/Human-Segmentation-Dataset.git




This project covers key concepts of data augmentations, creating datasets, and using trained weights from Imagenet for autoencoder for Unet-based architecture.

<h2>Key Packages Used</h2>
<ul>
  <li><b> CV2</b></li>
  <li><b> PyTorch</b></li>
  <li><b>NumPy</b> </li>
  <li><b>segmentation-models-pytorch</b> </li>
  <li> <b> Pandas</b></li>
  <li><b> Albumentations</b></li>
  <li><b>MatplotLib</b></li>
</ul>

<h2>Important Techniques</h2>
<ul>
  <li><b> Image Segmentation /Binary Segmentation </b></li>
  <li><b> Transfer Learning </b></li>
  <li><b> Data Augmentation</b></li>
  <li><b>Albumentations </b></li>
  <li><b> </b></li>
  </ul>

<h2><b>This project uses a human dataset, original author of the dataset is VikramShenoy97 and the data set is available: https://github.com/VikramShenoy97/Human-Segmentation-Dataset</b></h2>
