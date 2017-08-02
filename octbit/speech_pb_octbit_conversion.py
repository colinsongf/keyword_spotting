import tensorflow as tf

from tensorflow.python.platform import gfile
from plugins.lookahead import lookahead
import plugins.octbit.octbit_graph as qg
graph_def = tf.GraphDef()
with gfile.FastGFile("speech_graph.pb", 'rb') as f:
  graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as new_graph, tf.device("/cpu:0"):
  tf.import_graph_def(graph_def, name="")
  session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  rewriter = qg.GraphRewriter(session.graph_def, mode="octbit")
  a = rewriter.rewrite(['model/rnn_outputs', 'model/rnn_states', 'model/seq_lengths', 'model/ctc_decode_results', 'model/ctc_decode_log_probs'])
  tf.train.write_graph(
      a,
      ".",
      "octbit_speech.pb",
      as_text=False,
  )
  nodes = a.node
  nd = dict([(i.name, i) for i in nodes])
