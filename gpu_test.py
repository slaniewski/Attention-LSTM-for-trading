import tensorflow as tf

print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')

print("\n\n\n\nSYSCONFIG.GET_BUILD_INFO")
print(tf.sysconfig.get_build_info())
