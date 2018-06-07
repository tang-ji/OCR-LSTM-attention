import tensorflow as tf

from model import Model
from defaults import Config as defaults
tf.logging.set_verbosity(tf.logging.ERROR)

dataset_path = 'train.tfrecords'
num_epoch = 500
batch_size = 64
steps_per_checkpoint = 100
initial_learning_rate = 0.01

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    model = Model(
        phase='train',
        visualize=defaults.VISUALIZE,
        output_dir=defaults.OUTPUT_DIR,
        batch_size=batch_size,
        initial_learning_rate=initial_learning_rate,
        steps_per_checkpoint=steps_per_checkpoint,
        model_dir=defaults.MODEL_DIR,
        target_embedding_size=defaults.TARGET_EMBEDDING_SIZE,
        attn_num_hidden=128,
        attn_num_layers=8,
        clip_gradients=defaults.CLIP_GRADIENTS,
        max_gradient_norm=defaults.MAX_GRADIENT_NORM,
        session=sess,
        load_model=defaults.LOAD_MODEL,
        gpu_id=defaults.GPU_ID,
        use_gru=False,
        use_distance=defaults.USE_DISTANCE,
        max_image_width=defaults.MAX_WIDTH,
        max_image_height=defaults.MAX_HEIGHT,
        max_prediction_length=13,
        channels=1,
    )


    model.train(
        data_path=dataset_path,
        num_epoch=num_epoch
    )

    
    