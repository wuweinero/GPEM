import os
import tensorflow as tf
from panicle_net_class import PanicleNet

SAVE_PATH = './' #自行设置保存位置
NET_INPUT_SIZE = 80
net = PanicleNet(NET_INPUT_SIZE)

output_graph_def = tf.graph_util.convert_variables_to_constants(
    net.sess, net.sess.graph_def, output_node_names=['output'])
#形参output_node_names用于指定输出的节点名称

#路径一定要检查正确!!!!!!!!!!!
pb_path = os.path.join(SAVE_PATH, 'panicle_net_%d.pb' % NET_INPUT_SIZE)
#路径一定要检查正确!!!!!!!!!!!
with tf.gfile.FastGFile(pb_path, mode='wb') as f:
    f.write(output_graph_def.SerializeToString())
    print("save .pb successfully")
