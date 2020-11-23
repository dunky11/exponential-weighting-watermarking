# exponential-weighting-watermarking
This is an implemention of "[Robust Watermarking of Neural Network with Exponential Weighting](https://arxiv.org/pdf/1901.06151.pdf)"
 by Ryota Namba and Jun Sakuma in TensorFlow.

### What is exponential weighting?

Exponential weighting is the method which was proposed in the paper to make watermarks more robust against watermark removal attacks like pruning or fine-tuning. It works by applying a function to the weight matrix of each layer before it is used. The basic concept is:

1. Train the model on the training dataset until it converges
2. Enable exponential weighting in the layers of the model, so it first applies a transformation to the weight matrix before it is used in the forward pass
3. Train the model on the union of the key set and the training set in order to embed the watermark
4. Disable exponential weighting in the layers of the model

The key set can be any set of images. If the accuracy on the key set is above a predefined arbitrary threshold we can verify that the model belongs to us.
