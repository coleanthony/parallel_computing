import sys

def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian kernel for convolution."""

    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size+1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


if __name__ == "__main__":
    #size, mean, std = 2, 0.0, 1.5
    if (len(sys.argv) != 3):
        print("\nMakes 2D gaussian kernel for convolution\n")
        print("Requirements:\npython3, numpy, tensorflow, tensorflow-probability\n")
        print("Usage:\npython3 gaussian_kernel_calculator.py <blur radius> <standard deviation>\n")
    else:
        import numpy as np
        import tensorflow as tf
        import tensorflow_probability as tfp
        
        size, mean, std = int(sys.argv[1]), 0.0, float(sys.argv[2])
        gk = gaussian_kernel(size, mean, std)
        #sess = tf.Session()
        a = np.array(tf.Session().run(gk))
        np.savetxt('gk_r{0}_dev{1}.txt'.format(size, std), a, fmt='%.8f')
