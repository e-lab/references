# Deep Learning reading lists, 2011-present


## Courses:

Basic Machine Learning:
- https://www.coursera.org/course/ml
- https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/
- http://cs231n.stanford.edu/syllabus.html
- Purdue e_lab course: https://docs.google.com/document/d/1_p4Y_9Y79uBiMB8ENvJ0Uy8JGqhMQILIFrLrAgBXw60/edit#heading=h.ml4r2vcdki0v



## back-propagation

Theory / math:
http://www.deeplearningbook.org/contents/mlp.html
see chapter 6.5

linear layers:
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

conv net layers:
https://grzegorzgwardys.wordpress.com/2016/04/22/8/
and
https://www.slideshare.net/kuwajima/cnnbp
and
http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
[you need to save the maxpooling indices for back-prop]


## RNN - recurrent neural nets:

### nice explanation
https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

### RNN code is here:
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm


##RNN Recurrent neural networks useful links

Graphs in Torch: You need this before attempting to start digesting the LSTM code.
Exercise: https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf
Nando de Freitas lecture

Videos:
https://youtu.be/56TYLaQN4N8 (LSTM basics)
https://youtu.be/-yX1SYeDHbg (Alex Graves’s hand-writer algorithm)
Sides: https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/lecture11.pdf
Exercise: https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical6.pdf

Soumith article
http://devblogs.nvidia.com/parallelforall/understanding-natural-language-deep-neural-networks-using-torch/

Ilya Sutskever NIPS paper
Sequence to Sequence Learning with Neural Networks
http://arxiv.org/abs/1409.3215

Alex Graves (43 pages) paper
Generating Sequences With Recurrent Neural Networks
Prediction network – Long Short-term Memory Cell
Text Prediction
Penn Treebank Experiments
Wikipedia Experiments
Handwriting Prediction
Handwriting Synthesis
http://arxiv.org/abs/1308.0850

Alex Graves LSTM paper
Speech Recognition with Deep Recurrent Neural Networks
http://arxiv.org/abs/1303.5778

Wojciech Zaremba regularisation paper
Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

Hochreiter & Schmidhuber (the article)
http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Schmidhuber website
Plenty of applications and references
http://people.idsia.ch/~juergen/rnn.html

Schmidhuber LSTM tutorial
Video: https://www.youtube.com/watch?v=JSNZA8jVcm4
Slides: http://people.idsia.ch/~juergen/deep2014white.pdf

Andrej Karpathy blog post
Link: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

Understanding LSTM modules
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

And its uses:
http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

LSTM code torch explained by Adam:
https://apaszke.github.io/lstm-explained.html

## Augmented RNNs:

http://distill.pub/2016/augmented-rnns/

## Attention:

https://blog.heuritech.com/2016/01/20/attention-mechanism/


## Design neural nets:

VGG https://arxiv.org/pdf/1409.1556.pdf

PreLU http://arxiv.org/pdf/1603.05201v1.pdf

Inception v4 http://arxiv.org/abs/1602.07261

Inception v3 http://arxiv.org/abs/1409.4842

Inception v2 http://arxiv.org/abs/1512.00567

ResNet http://arxiv.org/abs/1512.03385

NiN http://arxiv.org/abs/1312.4400

systematic evaluation of modules:https://arxiv.org/abs/1606.02228

Xception: https://arxiv.org/abs/1610.02357

## Unsupervised

Soumith DCGAN
http://arxiv.org/abs/1511.06434

Solving Puzzles
http://arxiv.org/abs/1603.09246

co-occurence patches
https://arxiv.org/abs/1511.06811
http://graphics.cs.cmu.edu/projects/deepContext/

surrogate classes
http://arxiv.org/abs/1406.6909

video LSTM
http://arxiv.org/abs/1502.04681

learn to generate images from textual descriptions.
https://arxiv.org/abs/1605.05396


## sunsup / video prediciton

predicting next frames from video, MIT Torralba
http://arxiv.org/abs/1504.08023

prednet Coxlab: https://github.com/coxlab/prednet

Alf extra refs: https://docs.google.com/document/d/1_t7_Q4RxeX_blEQQXYv3MTDXxsCo2ZghfH7zp1y4Hkk

better frame reconstruction by predicting transformations:
https://arxiv.org/abs/1511.05440v6

Scrambling of video frames used to train unsup:
https://arxiv.org/pdf/1611.06646v2.pdf



# Reinforcement Learning

http://karpathy.github.io/2016/05/31/rl/


# 0-shot learning:

A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean, M. A.
Ranzato, and T. Mikolov. Devise: A deep visual-semantic
embedding model. In NIPS, 2013
https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf

R. Socher, M. Ganjoo, C. D. Manning, and A. Ng. Zero-shot
learning through cross-modal transfer. In NIPS. 2013.
https://nlp.stanford.edu/~socherr/SocherGanjooManningNg_NIPS2013.pdf

review:
https://arxiv.org/pdf/1703.04394.pdf


# 1-shot learning and adding new classes to pre-trained

http://fastml.com/good-representations-distance-metric-learning-and-supervised-dimensionality-reduction/

https://arxiv.org/pdf/1606.09282v2.pdf

https://arxiv.org/pdf/1606.04080.pdf


# transfer learning:

this paper on using CNN transfer learning ability to reach state-of-art in a lot of other dataset (transferred from ImageNet training):
http://arxiv.org/abs/1403.6382

This great paper also shows transfer from ImageNet to PASCAL VOC:
http://www.di.ens.fr/~josef/publications/oquab14.pdf

And this paper from Bengio group also has a great analysis:
https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf
https://arxiv.org/abs/1411.1792

This work on image segmentation also use transfer learning of VGG CNN networks:
http://arxiv.org/abs/1511.00561 

and more details here in this Stanford course material:
http://cs231n.github.io/transfer-learning/
http://cs231n.stanford.edu/reports2016/001_Report.pdf
http://cs231n.stanford.edu/reports2016/313_Report.pdf

transfer learning vs fully trained for vehicle model 
http://cs231n.stanford.edu/reports/lediurfinal.pdf

transfer learning from demonstrations for robot trajectory /LSTM + attentions
https://arxiv.org/abs/1703.07326

https://arxiv.org/abs/1707.03374

# segmentation:

https://arxiv.org/abs/1511.00561

https://arxiv.org/pdf/1505.04597v1.pdf

https://arxiv.org/pdf/1611.10080v1.pdf

https://arxiv.org/abs/1611.09326



# Artistic style
- https://arxiv.org/abs/1508.06576

# Super-resolution

- https://github.com/Tetrachrome/subpixel


## other paper / reference lists:

- https://github.com/terryum/awesome-deep-learning-papers
