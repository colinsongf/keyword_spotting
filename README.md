# keyword_spotting

Implement Chinese keyword spotting using RNN+CTC. This model is supposed to run on android phones or smaller devices, with low cpu and memory requirement.

training data: 230,000 speech wav, with text label, about 100 hrs

valid data: 1,024. positive-negtive half-half

keyword for experiment: 你好乐乐

basic structure: preprocessing -> rnn ->decode

##preprocessing

signal -> stft(linear spectrogram) -> mel spectrogram

in my experiment, we use **fft_size=25ms** and **hop_size=10ms** for stft, **n_mel=40** for mel filter bank with RNN **hidden_size=128**  is enough.

n\_mel=60 almost the same performance with n_mel=40. (My guess is that input feature size and RNN hidden size should match, and hidden\_size=128 is too small to model 60 feature from mel spectrogram.

Maybe larger hidden size and deeper network can perform better. But in our case, there is no need to use that large model.

I've tried mfcc, worse than mel.

##label

using CTC, the label is just text.

We use pinyin to represent words(using the marker), because some Chinese words have multiple phoneme, for example, one of our keyword 乐 has two pronounce: yue4 and le4, but we only want le4.

Our label space consist of keywords and garbage word(all other words except keyword)

We insert space between words in order to force the model to learn stronger ability to cut the words.

For example,

* 0 for space
* 1 for ni3
* 2 for hao3
* 3 for le3
* 4 for garbage word
* 5 for ctc blank

And therefore we have a output space of 6.

Sentence **你好乐乐** will be labeled as **010203030**

Actually we've tried frame-wise label with alignment for word(phoneme), but there is some difficulty and the outcome is not desirable. I will dicuss this in the loss function, later.

##model

RNN -> fully-connection layer -> CTC_loss(CTC_decode)

***Training model***

The model consist of 2 layer GRU RNN([http://arxiv.org/abs/1406.1078]()),with **hidden_size=128** ,no projection (have tried LSTM, no improvement, thus choosing GRU for less computation cost).

We have some optional wrapper for RNN Cell,

* Dropout wrapper
* Res wrapper (inspired by res net)
* Layernorm wrapper

However, in my experiment, the res wrapper and layernorm wrapper seems helpless. So I only use Dropout wrapper.(For dropout, I've tried [https://arxiv.org/abs/1512.05287](), which is called variational dropout, using the same mask within one recurrent layer, however the improvement is insignificant, and result in unstable performance. Therefore I use basic dropout.)

Fully-connection layer map the 128 hidden size to output space of num_classes = 6

We use CTC_loss as loss function, which do not require alignment and segment for label.
[http://www.cs.toronto.edu/~graves/icml_2006.pdf
]() Please notice that tf.ctc_loss will do the softmax for you, but for tf.ctc_beam_search_decode, you have to do softmax before feed into it.

I've tried cross-entropy-like loss function, and thus the data need to be labeled to frame-wise, i.e., the start and end of each word. The alignment and segment is done by our speech model, which, however, can only label the peak frame of each word instead of the boundry of the word (phoneme). And we can only give a rough approximation of the word boundry. We haven't found a perfect algorithm to do the alignment, thus the data quality is limited the to accuracy of the alignment. What's worse, it takes long time to do the alignment each time we want to add new data. With the cross-entropy loss, the model can only reach accurcary of about 85%.

***Deployment Model***

In the deployment model, we just replace the CTCloss with CTCdecoder to process the rnn_output (softmax before CTCdecode) to get the output sequence.

Actually, I write a simple decode function to replace the ctc_beam_search_decode, because our case is very simple, so there is no need to use ctc decode. More importantly, by doing this, we enable streaming on decode stage.

***Streaming for Deployment Model***

In the real production, we must enable streaming process, which mean we process the input audio in a almost real-time base, so as to reduce the latency.

The rnn itself can do streaming, and the decode function also support streaming.
**The key of streaming is to keep the rnn state.**


For example, we set a 300ms window size. Each time we feed 300ms audio into model, as well as the rnn state in the previous 300ms. And we fetch the softmax prob sequence as well as the rnn state.

We add a simple VAD function (based on volume) to detect voice activity.

A major drawback of this RNN model is that it will toally mess up when carrying the state of a long speech. (The reason, I guess, is that our training data are mostly short speech segment.) **So we clear the rnn state after each trigger and each non-speech segment detected by the VAD.** The VAD must be carefully tuned, otherwise it will cut off unfinished speech and clean the rnn state.

##Pipeline and Data

The data is about 80GB (linear spectrogram), which is too large to load into memory. So I save the data in tfrecords and feed into training model in streaming.

I maintain two queues to manage the pipeline, filename queue and data queue. This part is tricky so be careful if you want to hack this part fo code, otherwise the pipeline might be stuck. Or if can also use tf's build-in input queue.(My queue is similar to tf's own queue, but add some customized features)

You can choose whatever data you want to save in tfrecords, raw wave or linear spectrogram or mel spectrogram. With respect to data volume, **linear spec>raw wave>mel spec** According to my experience, the computation of preprocessing is insignificant.

**My advice is to save raw audio or linear spectrogram in tfrecords**, because it's much easier to do some data enhancement (for example, adding noise, or other trick as you like) on the fly with linear domain, once it becomes mel spectrogram, things get much more complicated.

One more thing I would like to mention: the usage of tfrecord is also tricky, and the document seems missing from tf office website. So also be careful if you want to modify this part. A good practise can be found here: [http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
]()

Also the ctc function require sparse tensor as label, please be familiar with the operation of sparse tensor.

##Customize keyword

In the customize branch, I implement a new feature which enable customized keyword by recording only 3 speech utterances. The new model is trained on original pre-trained model, with few data and fast training.

This part is still in experiment.

**The basic idea:**

The rnn part only learn to extract the speech features, so basically it has little to do with the output space projection.

Therefore, we only want to modify the fully-connection matrix weights, and freezing all other variables.

For our expeirment keyword 你好乐乐, we have a [128,6] weights matrix, where hidden\_size = 128 and num\_classes = 6. To add new customized keyword, for example, 苹果, we add a new [128,2] weight matrix and concat them.

* [128,4] original weights, for (space,ni3,hao3,le4) **trainable=False**
* [128,1] original weights, for (garbage word) **trainable=False**
* [128,2] new weights, for (ping2, guo3) **trainable=True**
* [128,1] original weights, for (ctc blank) **trainable=False**

Theoritically, the new keyword is included in original garbage words, so if we want to modify the origin garbage words weights to add new mappings, we have to train the garbage words weights as well. However, the problem is that we want to train the new with only a few data, and the garbage words weight will totally mess up due to lack of adequate negtive data.

The ideal way is to train the garbage words weight but with as less change as possible. But I haven't figure out a way to do this, so I just freeze this weights and train the new weigihts with a few positive samples. The accurcay is acceptable.

Another problem is that logits scale of RNN outputs is not comparable between old weights and new weights. When doing softmax, this might cause problem, for example, the original keyword ni3 will be recognized as ping2. Still, I haven't figure out a way to fix this. The strategy I use now is to keep two fully-connection matrix, i.e., [128,6] and [128,8], and do softmax and decode respectively.


##Attention

Inspired by [https://arxiv.org/abs/1706.03762
](), I've tried to use self-Attention to replace RNN in the model structure, with other parts unchange.

A significant advantage of attention is fast training, thanks to parallel computation. The accurcay is almost the same as GRU RNN(slightly better).

But attention model doesn't support streaming, so we only use it for experiment.

Some notes for attention:

* it's highly sentive to dropout rate. In my experient, keep\_prob=0.9 give the best result, keep\_prob<0.7 will totally mess up. I'm not sure about the reason.
* sentive to noise level added in training data.
* sentive to learning rate.

Given that keywords is short speech utterance and we process windowed streaming input, this might work in read production, potentially. Still need further experiment to verify this.







