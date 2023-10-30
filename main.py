#!/usr/bin/env python
# -*- coding: utf-8 -*-
import src.data as pr
import src.model as mod
import src.visualization as vis

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

def main():

    # Preprocessing
    # prep = pr.TrainPrep()

    # #prep.prep_icsa()

    # prep.prep_tis()

    # prep.join_inputs()

    # # Data split into train-test windows
    # ws = pr.WindowSplit()

    # ws.generate_windows()

    # # # Fit, Predict, save predictions
    # fp = mod.RollingLSTM()

    # fp.model_fit_predict_multiprocess()

    # fp.save_results()


    # # # # Get performance metrics
    metrics = mod.PerformanceMetrics()

    metrics.load_latest_eval_data()

    metrics.calculate_metrics()


    # # Visualize results
    vs = vis.Plots()

    vs.load_latest_data()

    vs.hist()

    vs.equity_line()

    return 0


if __name__ == "__main__":
    main()

    ### If you have issues with GPU... Good luck! This code might help debug.
    ##pip uninstall tensorflow
    ##pip install tensorflow-gpu
    #print(tf.__version__)
    #print(tf.sysconfig.get_build_info())
    #if tf.test.is_gpu_available():
    #    print("TensorFlow GPU version is installed.")

    #with tf.device('/GPU:0'):
    #    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    #    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    #    c = tf.matmul(a, b)
    #print(c)


    #else:
    #    print("TensorFlow GPU version is NOT installed.")
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.set_visible_devices(gpus[0], 'GPU')
    #try:
    ##    # Disable all GPUS
    ##    tf.config.set_visible_devices([], 'GPU')
    ##    visible_devices = tf.config.get_visible_devices()
    ##    for device in visible_devices:
    ##        assert device.device_type != 'GPU'
    ##except:
    #    # Invalid device or cannot modify virtual devices once initialized.
    #    pass