# Sequence-to-sequence modeling for NLP

Another way to solve similar task

* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
    > General end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. The method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. The experimentation part tackles an English to French translation task from the WMT’14 dataset.


* [Grammar as a Foreign Language](https://arxiv.org/pdf/1412.7449.pdf)
    > This paper tackles the problem of syntactic constituency parsing in NLP. It shows how the domain agnostic attention-enhanced sequence-to-sequence model achieves state-of-the-art results on the most widely used syntactic constituency parsing dataset.

* [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf)
    > Sequence to sequence architecture based entirely on convolutional neural networks. Compared to recurrent models, computations over all elements can be fully parallelized during training to better exploit the GPU hardware and optimization is easier since the number of non-linearities is fixed and independent of the input length. A gated linear units approach eases gradient propagation and equipping each decoder layer with a separate attention module outperforms the accuracy of commun translation tasks.

* [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869v1.pdf)
    > A simple approach for conversational modeling which uses the recently proposed sequence to sequence framework. The model converses by predicting the next sentence given the  previous sentence or sentences in a conversation. The strength of the model is that it can be trained  end-to-end and thus requires much fewer hand-crafted rules. This straightforward model can generate simple conversations given a large conversational training dataset. It is able to extract knowledge from both a domain specific dataset, and from a large, noisy, and general domain dataset of movie subtitles. On a domain-specific IT helpdesk dataset, the model can find a solution to a technical problem via conversations. On a noisy open-domain movie transcript dataset, the model can perform simple forms of common sense reasoning.

* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
    > A novel neural network model called RNN Encoder–Decoder that consists of two recurrent neural networks (RNN). One RNN encodes a sequence of symbols into a fixed-length vector representation, and the other decodes the representation into another sequence of symbols. The encoder and decoder of the proposed model are jointly trained to maximize the conditional probability of a target sequence given a source sequence. The performance of a statistical machine translation system is empirically found to improve by using the conditional probabilities of phrase pairs computed by the RNN Encoder–Decoder as an additional feature in the existing log-linear model. Qualitatively  we show that the proposed model learns a semantically and syntactically meaningful representation of linguistic phrases.

* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
    > Unlike the traditional statistical machine translation,the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder–decoders and encode a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, it is conjectured that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder–decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly.

# Other neural language models

* [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)
    > A simple neural language model that relies only on character-level inputs. Predictions are still made at the  word-level. The model employs a CNN and a highway network over characters, whose output is given to a LSTM RNN language model. On the English Penn Treebank the model is on par with the existing state-of-the-art  despite  having 60% fewer parameters. On languages with rich morphology (Arabic, Czech, French, German, Spanish, Russian), the model outperforms word-level/morpheme-level LSTM baselines,again with fewer parameters. The results suggest that on many languages, character inputs are sufficient for language modeling. Analysis of word representations obtained from the character composition part of the model reveals that the model is able to encode, from characters only, both semantic and orthographic information.
