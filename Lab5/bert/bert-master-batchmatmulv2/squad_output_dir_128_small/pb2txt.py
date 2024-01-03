# This file is useful for reading the contents of the ops generated by ruby.
# You can read any graph defination in pb/pbtxt format generated by ruby
# or by python and then convert it back and forth from human readable to binary format.

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import sys

if len(sys.argv) != 3:
    print("{} pb_path pbtxt_path".format(sys.argv[0]))
pb_path= sys.argv[1]
pbtxt_path = sys.argv[2]

def graphdef_to_pbtxt(filename,pbtxt_path):
    with gfile.FastGFile(filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', pbtxt_path, as_text=True)
    return

graphdef_to_pbtxt(pb_path,pbtxt_path)  