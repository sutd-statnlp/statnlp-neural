# StatNLP-PyTorch (V0.1)
The PyTorch version of StatNLP implementation V0.1
* [1. Requirements and Installation](#requirements-and-installation)
* [2. Examples](#examples)
* [3. Implementation Tutorials](#implementation-tutorials)

## Requirements and Installation
The project is based on PyTorch 0.4+ and Python 3.5+.

We plan to upload our framework to Pypi where you can use the framework by simply typing `pip install`. But at the moment, what you can do is:
```bash
git clone https://github.com/leodotnet/statnlp-neural
```
Build your neural graphical model under this code base.

## Examples
We have built some existing models with this framework for your references:
* [Linear CRF for Named Entity Recognition](/examples/linearner.py)
* [Semi-Markov CRF for NP Chunking](/examples/semi.py)
* [CNN for Text Classification](/examples/textclass.py)
* [Constituency Parsing CRF Model](/examples/textclass.py)

## Implementation Tutorials

Get your hands dirty with StatNLP! If you are not familiar with the fundamental theory of StatNLP, check out our [EMNLP 2017 tutorial on structured prediction](http://www.statnlp.org/tutorials).

Follow the tutorials below to build your model.
1. [Basics: load data into instances](/docs/basics.md)
2. [Graphical Model: build customized graphical model](/docs/graphs.md)
3. [Neural Network: design neural network](/docs/neural.md)
4. [Run the model!](/docs/run.md)

## Contributing

Coming soon

## Contact

Please email to [Li Hao](http://www.statnlp.org/graduate-students/li_hao) and [Allan](https://people.sutd.edu.sg/~allanjie/) for suggestions and comments.


## License
GNU general public