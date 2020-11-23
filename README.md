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

### How to use

You can create your own exponentially weighted layers by inheriting from [EWBase](https://github.com/dunky11/exponential-weighting-watermarking/blob/main/ew.py) which inherits from keras.layers.Layer. If exponential weighting is enabled, just call EWBase.ew() on the weight matrix before using it in the forward pass of your layer.

A simple example can be found in [example.ipynb](https://github.com/dunky11/exponential-weighting-watermarking/blob/main/example.ipynb) or [example.py](https://github.com/dunky11/exponential-weighting-watermarking/blob/main/example.py). 


## Contribute
Show your support by ⭐ the project. Pull requests are always welcome.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/dunky11/react-saas-template/blob/master/LICENSE) file for details.
