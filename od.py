# Dependencies
import tensorflow as tf
import numpy as np


# load graphs using pb file path
def load_graph(pb_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


# returns tensor dictionaries from graph
def get_inference(graph, count=0):
    with graph.as_default():
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks', 'image_tensor']:
            tensor_name = key + ':0' if count == 0 else '_{}:0'.format(count)
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().\
                                        get_tensor_by_name(tensor_name)
        return tensor_dict


# renames while_context because there is one while function for every graph
# open issue at https://github.com/tensorflow/tensorflow/issues/22162
def rename_frame_name(graphdef, suffix):
    for n in graphdef.node:
        if "while" in n.name:
            if "frame_name" in n.attr:
                n.attr["frame_name"].s = str(n.attr["frame_name"]).replace("while_context",
                                                                           "while_context" + suffix).encode('utf-8')


if __name__ == '__main__':

    # your pb file paths
    frozenGraphPath1 = 'frozen_graphs/human_frozen_inference_graph.pb'
    frozenGraphPath2 = 'frozen_graphs/mask_frozen_inference_graph.pb'

    # new file name to save combined model
    combinedFrozenGraph = 'combined_frozen_inference_graph.pb'

    # loads both graphs
    graph1 = load_graph(frozenGraphPath1)
    graph2 = load_graph(frozenGraphPath2)

    # get tensor names from first graph
    tensor_dict1 = get_inference(graph1)

    with graph1.as_default():

        # getting tensors to add crop and resize step
        image_tensor = tensor_dict1['image_tensor']
        scores = tensor_dict1['detection_scores'][0]
        num_detections = tf.cast(tensor_dict1['num_detections'][0], tf.int32)
        detection_boxes = tensor_dict1['detection_boxes'][0]

        # I had to add NMS becuase my ssd model outputs 100 detections and hence it runs out of memory becuase of huge tensor shape
        selected_indices = tf.image.non_max_suppression(detection_boxes, scores, 5, iou_threshold=0.5)
        selected_boxes = tf.gather(detection_boxes, selected_indices)

        # intermediate crop and resize step, which will be input for second model(FRCNN)
        cropped_img = tf.image.crop_and_resize(image_tensor,
                                               selected_boxes,
                                               tf.zeros(tf.shape(selected_indices), dtype=tf.int32),
                                               [300, 60] # resize to 300 X 60
                                               )
        cropped_img = tf.cast(cropped_img, tf.uint8, name='cropped_img')


    gdef1 = graph1.as_graph_def()
    gdef2 = graph2.as_graph_def()

    g1name = "graph1"
    g2name = "graph2"

    # renaming while_context in both graphs
    rename_frame_name(gdef1, g1name)
    rename_frame_name(gdef2, g2name)

    # This combines both models and save it as one
    with tf.Graph().as_default() as g_combined:

        x, y = tf.import_graph_def(gdef1, return_elements=['image_tensor:0', 'cropped_img:0'])

        z, = tf.import_graph_def(gdef2, input_map={"image_tensor:0": y}, return_elements=['detection_boxes:0'])

        tf.train.write_graph(g_combined, "./frozen_graphs", combinedFrozenGraph, as_text=False)