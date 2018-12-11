#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
# import cPickle as pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.skin.data_utils import read_data
from src.skin.general_controller import GeneralController
from src.skin.general_child import GeneralChild

from src.skin.micro_controller import MicroController
from src.skin.micro_child import MicroChild

flags = tf.app.flags
FLAGS = flags.FLAGS

# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_string("search_for", None, "Must be [macro|micro]")

DEFINE_integer("batch_size", 32, "")

DEFINE_integer("output_classes", 5, "")
DEFINE_integer("img_size", 224, "")
DEFINE_integer("num_epochs", 300, "")
DEFINE_integer("child_lr_dec_every", 30, "")
DEFINE_integer("child_num_layers", 5, "")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 48, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 4, "")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_block_size", 3, "")
DEFINE_integer("child_lr_T_0", None, "for lr schedule")
DEFINE_integer("child_lr_T_mul", None, "for lr schedule")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.01, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.5, "")
DEFINE_float("child_drop_path_keep_prob", 1.0, "minimum drop_path_keep_prob")
DEFINE_float("child_l2_reg", 1e-4, "")
DEFINE_float("child_lr_max", None, "for lr schedule")
DEFINE_float("child_lr_min", None, "for lr schedule")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_op_tanh_reduce", 1.0, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", False, "")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")
DEFINE_boolean("controller_use_critic", False, "")
DEFINE_boolean("controller_from_fixed", False, "") # search architecture starting from fixed architecture

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")


def get_ops(datasets, shapes):
    """
    Args:
      images: dict with keys {"train", "valid", "test"}.
      labels: dict with keys {"train", "valid", "test"}.
    """

    # micro or macro
    assert FLAGS.search_for is not None, "Please specify --search_for"

    if FLAGS.search_for == "micro":
        ControllerClass = MicroController
        ChildClass = MicroChild
    else:
        ControllerClass = GeneralController
        ChildClass = GeneralChild

    child_model = ChildClass(
        datasets,
        shapes,
        use_aux_heads=FLAGS.child_use_aux_heads, # question: whether or not use auxiliary network for predict
        cutout_size=FLAGS.child_cutout_size, # randomly set one or more positions to black
        whole_channels=FLAGS.controller_search_whole_channels,
        num_layers=FLAGS.child_num_layers,
        num_cells=FLAGS.child_num_cells,
        num_branches=FLAGS.child_num_branches,
        fixed_arc=FLAGS.child_fixed_arc,
        out_filters_scale=FLAGS.child_out_filters_scale,
        out_filters=FLAGS.child_out_filters,
        keep_prob=FLAGS.child_keep_prob,
        drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
        num_epochs=FLAGS.num_epochs,
        l2_reg=FLAGS.child_l2_reg,
        data_format=FLAGS.data_format,
        batch_size=FLAGS.batch_size,
        clip_mode="norm",
        grad_bound=FLAGS.child_grad_bound,
        lr_init=FLAGS.child_lr,
        lr_dec_every=FLAGS.child_lr_dec_every,
        lr_dec_rate=FLAGS.child_lr_dec_rate,
        lr_cosine=FLAGS.child_lr_cosine,
        lr_max=FLAGS.child_lr_max,
        lr_min=FLAGS.child_lr_min,
        lr_T_0=FLAGS.child_lr_T_0,
        lr_T_mul=FLAGS.child_lr_T_mul,
        optim_algo="momentum",
        sync_replicas=FLAGS.child_sync_replicas,
        num_aggregate=FLAGS.child_num_aggregate,
        num_replicas=FLAGS.child_num_replicas,
        output_classes=FLAGS.output_classes
    )

    if (FLAGS.child_fixed_arc is None) or FLAGS.controller_from_fixed:
        '''
        if child architecture is not fixed or we want to search child architecture from a fixed arc, controller need to be trained.
        '''
        controller_model = ControllerClass(
            search_for=FLAGS.search_for,
            search_whole_channels=FLAGS.controller_search_whole_channels,
            skip_target=FLAGS.controller_skip_target,
            skip_weight=FLAGS.controller_skip_weight,
            num_cells=FLAGS.child_num_cells,
            num_layers=FLAGS.child_num_layers,
            num_branches=FLAGS.child_num_branches,
            out_filters=FLAGS.child_out_filters,
            lstm_size=64,
            lstm_num_layers=1,
            lstm_keep_prob=1.0,
            tanh_constant=FLAGS.controller_tanh_constant,
            op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
            temperature=FLAGS.controller_temperature,
            lr_init=FLAGS.controller_lr,
            lr_dec_start=0,
            lr_dec_every=1000000,  # never decrease learning rate
            l2_reg=FLAGS.controller_l2_reg,
            entropy_weight=FLAGS.controller_entropy_weight,
            bl_dec=FLAGS.controller_bl_dec,
            use_critic=FLAGS.controller_use_critic,
            optim_algo="adam",
            sync_replicas=FLAGS.controller_sync_replicas,
            num_aggregate=FLAGS.controller_num_aggregate,
            num_replicas=FLAGS.controller_num_replicas)

        print('Connecting...')
        child_model.connect_controller(controller_model)
        print('Building trainer...')
        controller_model.build_trainer(child_model)

        controller_ops = {
            "train_step": controller_model.train_step,
            "loss": controller_model.loss,
            "train_op": controller_model.train_op,
            "lr": controller_model.lr,
            "grad_norm": controller_model.grad_norm,
            "valid_acc": controller_model.valid_acc,
            "optimizer": controller_model.optimizer,
            "baseline": controller_model.baseline,
            "entropy": controller_model.sample_entropy,
            "sample_arc": controller_model.sample_arc,
            "skip_rate": controller_model.skip_rate,
        }
    else:
        assert not FLAGS.controller_training, (
            "--child_fixed_arc is given, cannot train controller")
        child_model.connect_controller(None)
        controller_ops = None

    child_ops = {
        "global_step": child_model.global_step,
        "loss": child_model.loss,
        "train_op": child_model.train_op,
        "lr": child_model.lr,
        "grad_norm": child_model.grad_norm,
        "train_acc": child_model.train_acc,
        "optimizer": child_model.optimizer,
        "num_train_batches": child_model.num_train_batches,
    }

    ops = {
        "child": child_ops,
        "controller": controller_ops,
        "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
        "eval_func": child_model.eval_once,
        "num_train_batches": child_model.num_train_batches,
    }

    return ops


def generate_data():
    # this function is just for debugging
    images, labels = {}, {}
    size = 100
    num1 = 10
    num2 = 10
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            images[key] = np.random.random_sample((num1, size, size, 3)).astype('float32')
            labels[key] = np.array(range(num1)).astype('int32')
        else:
            images[key] = np.random.random_sample((num2, size, size, 3)).astype('float32')
            labels[key] = np.array(range(num2)).astype('int32')
        print('{}: img.shape:{} labels.shape:{}'.format(key, images[key].shape, labels[key].shape))
    return images, labels

def bd(x, y, batch_size=4, isTrain=False):
    # build_dataset
    return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).batch(batch_size).repeat()

def train():
    if FLAGS.child_fixed_arc is None:
        images, labels = read_data(FLAGS.data_path, FLAGS.img_size)
    else:
        images, labels = read_data(FLAGS.data_path, FLAGS.img_size, num_valids=0)
    # images, labels = generate_data()

    images_valid_rl = np.copy(images['valid'])
    labels_valid_rl = np.copy(labels['valid'])
    if FLAGS.data_format == 'NCHW':
        for key in images:
            images[key] = np.transpose(images[key], [0, 3, 1, 2])

    train_num = images['train'].shape[0]
    valid_num = images['valid'].shape[0]
    test_num = images['test'].shape[0]
    
    shapes = {
        'train': train_num,
        'valid': valid_num,
        'test': test_num,
        'img_size': images['train'].shape[2] 
    }

    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('input'):
            x_train = tf.placeholder(images['train'].dtype, images['train'].shape, 'x_train')
            y_train = tf.placeholder(labels['train'].dtype, labels['train'].shape, 'y_train')
            x_valid = tf.placeholder(images['valid'].dtype, images['valid'].shape, 'x_valid')
            y_valid = tf.placeholder(labels['valid'].dtype, labels['valid'].shape, 'y_valid')
            x_test = tf.placeholder(images['test'].dtype, images['test'].shape, 'x_test')
            y_test = tf.placeholder(labels['test'].dtype, labels['test'].shape, 'y_test')
            x_valid_rl = tf.placeholder(images['valid'].dtype, images['valid'].shape, 'x_valid_rl')
            y_valid_rl = tf.placeholder(labels['valid'].dtype, labels['valid'].shape, 'y_valid_rl')
            
        datasets = {}
        train_dataset = bd(x_train, y_train, batch_size=FLAGS.batch_size, isTrain=True)
        valid_dataset = bd(x_valid, y_valid, batch_size=FLAGS.batch_size, isTrain=False)
        valid_rl_dataset = bd(x_valid_rl, y_valid_rl, batch_size=FLAGS.batch_size, isTrain=False)
        test_dataset = bd(x_test, y_test, batch_size=FLAGS.batch_size, isTrain=False)

        train_iterator = train_dataset.make_initializable_iterator()
        valid_iterator = valid_dataset.make_initializable_iterator()
        valid_rl_iterator = valid_rl_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()

        datasets['train'] = train_iterator.get_next() # return x_train_batch, y_train_batch
        datasets['valid'] = valid_iterator.get_next()
        datasets['test']  = test_iterator.get_next()
        datasets['valid_rl']  = valid_rl_iterator.get_next()

        def feed_dict(flag='train'):
            if flag == 'train':
                return {x_train: images['train'], y_train: labels['train']}
            elif flag == 'valid':
                return {x_valid: images['valid'], y_valid: labels['valid']}
            elif flag == 'test':
                return {x_test: images['test'], y_test: labels['test']}
            elif flag == 'valid_rl':
                return {x_valid_rl: images_valid_rl, y_valid_rl: labels_valid_rl}

        ops = get_ops(datasets, shapes)
        child_ops = ops["child"]
        controller_ops = ops["controller"]

        saver = tf.train.Saver(max_to_keep=2)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)

        hooks = [checkpoint_saver_hook]
        if FLAGS.child_sync_replicas:
            sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(
                True)
            hooks.append(sync_replicas_hook)
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
            sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(
                True)
            hooks.append(sync_replicas_hook)

        print("-" * 80)
        print("Starting session")
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
            start_time = time.time()

            sess.run(train_iterator.initializer, feed_dict=feed_dict('train'))
            sess.run(valid_iterator.initializer, feed_dict=feed_dict('valid'))
            sess.run(test_iterator.initializer, feed_dict=feed_dict('test'))
            sess.run(valid_rl_iterator.initializer, feed_dict=feed_dict('valid_rl'))

            train_acc = 0.0
            while True:
                #####################################
                ######  calculate child ops  ########
                #####################################

                run_ops = [
                    child_ops["loss"],
                    child_ops["lr"],
                    child_ops["grad_norm"],
                    child_ops["train_acc"],
                    child_ops["train_op"],
                ]
                loss, lr, gn, tr_acc, _ = sess.run(run_ops)
                global_step = sess.run(child_ops["global_step"]) # start from 1
                
                if FLAGS.child_sync_replicas:
                    actual_step = global_step * FLAGS.num_aggregate
                else:
                    actual_step = global_step

                # ops["num_train_batches"] stands for N steps/epoch, "epoch" stands for the current epoch
                epoch = actual_step // ops["num_train_batches"] # start from 0
                curr_time = time.time()

                num_batches_for_one_epoch = shapes['train']//FLAGS.batch_size
                train_acc += tr_acc

                if actual_step % num_batches_for_one_epoch == 0:
                    print("Epoch:{:<6d} train_acc:{}".format(epoch, train_acc/shapes['train']))
                    train_acc = 0

                if global_step % FLAGS.log_every == 0:
                    log_string = "Child: "
                    log_string += "epoch={:<6d}".format(epoch) # the number of current epoch
                    log_string += " global_step={:<6d}".format(global_step)
                    log_string += " loss={:<8.6f}".format(loss)
                    log_string += " lr={:<8.4f}".format(lr)
                    log_string += " |g|={:<8.4f}".format(gn)
                    log_string += " train_acc={:<3d}/{:>3d}".format(
                        tr_acc, FLAGS.batch_size) # right numbers/batch size
                    log_string += " mins={:<10.2f}".format(
                        float(curr_time - start_time) / 60)
                    print(log_string)
                
                #########################################
                ###### calculate controller ops   #######
                #########################################
                
                if actual_step % ops["eval_every"] == 0:
                    if (FLAGS.controller_training and
                            epoch % FLAGS.controller_train_every == 0):
                        print("Epoch {}: Training controller".format(epoch))
                        for ct_step in range(FLAGS.controller_train_steps *
                                             FLAGS.controller_num_aggregate):
                            run_ops = [
                                controller_ops["loss"],
                                controller_ops["entropy"],
                                controller_ops["lr"],
                                controller_ops["grad_norm"],
                                controller_ops["valid_acc"],
                                controller_ops["baseline"],
                                controller_ops["skip_rate"],
                                controller_ops["train_op"],
                            ]
                            
                            loss, entropy, lr, gn, val_acc, bl, skip, _ = sess.run(run_ops)
                            controller_step = sess.run(
                                controller_ops["train_step"])
                
                            if ct_step % FLAGS.log_every == 0:
                                curr_time = time.time()
                                log_string = "Controller: "
                                log_string += "ctrl_step={:<6d}".format(
                                    controller_step)
                                log_string += " loss={:<7.3f}".format(loss)
                                log_string += " entropy={:<5.2f}".format(entropy)
                                log_string += " lr={:<6.4f}".format(lr)
                                log_string += " |g|={:<8.4f}".format(gn)
                                log_string += " valid_acc={:<6.4f}".format(val_acc)
                                log_string += " baseline={:<5.2f}".format(bl)
                                log_string += " mins={:<.2f}".format(
                                    float(curr_time - start_time) / 60)
                                print(log_string)
                
                        print("Here are 10 architectures")
                        for _ in range(10):
                            arc, acc = sess.run([
                                controller_ops["sample_arc"],
                                controller_ops["valid_acc"],
                            ], feed_dict=feed_dict('valid'))
                            if FLAGS.search_for == "micro":
                                normal_arc, reduce_arc = arc
                                print('Normal_arc:{}'.format(np.reshape(normal_arc, [-1])))
                                print('Reduce_arc:{}'.format(np.reshape(reduce_arc, [-1])))
                            else:
                                start = 0
                                for layer_id in range(FLAGS.child_num_layers):
                                    if FLAGS.controller_search_whole_channels:
                                        end = start + 1 + layer_id
                                    else:
                                        end = start + 2 * FLAGS.child_num_branches + layer_id
                                    print(np.reshape(arc[start: end], [-1]))
                                    start = end
                            print("controller_valid_acc={:<6.4f}".format(acc))
                            print("-" * 80)
                
                    print("Epoch {}: Eval".format(epoch))
                    if FLAGS.child_fixed_arc is None:
                        ops["eval_func"](sess, "valid")
                    ops["eval_func"](sess, "test")
                
                if epoch >= FLAGS.num_epochs:
                    break


def main(_):
    print("-" * 80)
    if not os.path.isdir(FLAGS.output_dir):
        print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)

    print("-" * 80)
    log_file = os.path.join(FLAGS.output_dir, "stdout")
    print("Logging to {}".format(log_file))
    sys.stdout = Logger(log_file)

    utils.print_user_flags()
    train()


if __name__ == "__main__":
    tf.app.run()
