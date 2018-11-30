# Environment Requirements

Python2.7 + TensorFlow-gpu 1.4.0

**Very Very Very important！！！！！！！！**

-----

# Changes

My own dataset actually is larger than cifar10 and mnist, because my image size is normally over 1000*500. If I scale the image, I can only set the set the image size 160\*160. Otherwise, it will raise `ValueError: GraphDef cannot be larger than 2GB`, which confused me for nearly 3 weeks.... so sad. But now, I've modified the original code to load my own dataset successfully. The following are the details:

- modify the file: **data_utils.py**. There two ways of loading data. 
  - Load data using `torchvision`. This method is a little time consuming, but you can try different image size easily and select a proper size.
  - Only load data once, and then save the data to `pkl` file. 
- The resaon why the code raise `ValueError: GraphDef cannot be larger than 2GB`, in my opinion, is that original code load the whole data for once. And the data will be transfered to `Tensor` or maybe `Constant`, which will take up lots of graph space. In order to solve this problem, I use tf.placeholder. 

Many thanks to these websites:
- [Tensorflow: create minibatch from numpy array > 2 GB](https://stackoverflow.com/questions/49053569/tensorflow-create-minibatch-from-numpy-array-2-gb)
- [TensorFlow: does tf.train.batch automatically load the next batch when the batch has finished training?](https://stackoverflow.com/questions/41673889/tensorflow-does-tf-train-batch-automatically-load-the-next-batch-when-the-batch)

I have to say, TensorFlow is really really hard to use especially for the new people, just like me. Although I've solved the problem of loading data, I still have the problem to modify the code to run on multi GPUs. So if you are also interested in or you are stuck in this problem, welcome to join me or pull your requests.

# How to load your datasets

There are several things you need to do:
- Change the bash script for you own dataset. see `skin5_placeholder_micro_search.sh`.
- change the imported module name.
> (These files need to be modified: `main.py, micro_child.py, models.py, general_child.py`)

![](https://ask.qcloudimg.com/draft/1215004/rdhgi8s3yg.png)
- It's optional to change the image size in `data_utils.py`.

# Efficient Neural Architecture Search via Parameter Sharing

Authors' implementation of "Efficient Neural Architecture Search via Parameter Sharing" (2018) in TensorFlow.

Includes code for CIFAR-10 image classification and Penn Tree Bank language modeling tasks.

Paper: https://arxiv.org/abs/1802.03268

Authors: Hieu Pham*, Melody Y. Guan*, Barret Zoph, Quoc V. Le, Jeff Dean

_This is not an official Google product._

## Penn Treebank

The Penn Treebank dataset is included at `data/ptb`. Depending on the system, you may want to run the script `data/ptb/process.py` to create the `pkl` version. All hyper-parameters are specified in these scripts.

To run the ENAS search process on Penn Treebank, please use the script
```
./scripts/ptb_search.sh
```

To run ENAS with a determined architecture, you have to specify the archiecture using a string. The following is an example script for using the architecture we described in our paper.
```
./scripts/ptb_final.sh
```
A sequence of architecture for a cell with `N` nodes can be specified using a sequence `a` of `2N + 1` tokens

* `a[0]` is a number in `[0, 1, 2, 3]`, specifying the activation function to use at the first cell: `tanh`, `ReLU`, `identity`, and `sigmoid`.
* For each `i`, `a[2*i]` specifies a previous index and `a[2*i+1]` specifies the activation function at the `i`-th cell.

For a concrete example, the following sequence specifies the architecture we visualize in our paper

```
0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1
```

<img src="https://github.com/melodyguan/enas/blob/master/img/enas_rnn_cell.png" width="50%"/>

## CIFAR-10

To run the experiments on CIFAR-10, please first download the [dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Again, all hyper-parameters are specified in the scripts that we descibe below.

To run the ENAS experiments on the _macro search space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_macro_search.sh
./scripts/cifar10_macro_final.sh
```

A macro architecture for a neural network with `N` layers consists of `N` parts, indexed by `1, 2, 3, ..., N`. Part `i` consists of:

* A number in `[0, 1, 2, 3, 4, 5]` that specifies the operation at layer `i`-th, corresponding to `conv_3x3`, `separable_conv_3x3`, `conv_5x5`, `separable_conv_5x5`, `average_pooling`, `max_pooling`.
* A sequence of `i - 1` numbers, each is either `0` or `1`, indicating whether a skip connection should be formed from a the corresponding past layer to the current layer.

A concrete example can be found in our script `./scripts/cifar10_macro_final.sh`.

To run the ENAS experiments on the _micro search space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_micro_search.sh
./scripts/cifar10_micro_final.sh
```

A micro cell with `B + 2` blocks can be specified using `B` blocks, corresponding to blocks numbered `2, 3, ..., B+1`, each block consists of `4` numbers
```
index_1, op_1, index_2, op_2
```
Here, `index_1` and `index_2` can be any previous index. `op_1` and `op_2` can be `[0, 1, 2, 3, 4]`, corresponding to `separable_conv_3x3`, `separable_conv_5x5`, `average_pooling`, `max_pooling`, `identity`.

A micro architecture can be specified by two sequences of cells concatenated after each other, as shown in our script `./scripts/cifar10_micro_final.sh`

## Citations

If you happen to use our work, please consider citing our paper.
```
@inproceedings{enas,
  title     = {Efficient Neural Architecture Search via Parameter Sharing},
  author    = {Pham, Hieu and
               Guan, Melody Y. and
               Zoph, Barret and
               Le, Quoc V. and
               Dean, Jeff
  },
  booktitle = {ICML},
  year      = {2018}
}
```
