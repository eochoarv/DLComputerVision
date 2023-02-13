"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        stride = conv_param['stride']
        pad = conv_param['pad']

        H_ = int(1 + (H + 2 * pad - HH) / stride)
        W_ = int(1 + (W + 2 * pad - WW) / stride)
        
        x_ = x.new_zeros((N, C, H + 2*pad, W + 2*pad))
        x_[:, :,pad:-pad,pad:-pad] = x

        out = x.new_zeros((N, F, H_, W_))

        for f in range(F):
          for n in range(N):
            for h in range(H_):
              for w_ in range(W_):
                out[n,f,h,w_] = (x_[n,:,h*stride:(h*stride+HH),w_*stride:(w_*stride+WW)] * w[f]).sum() + b[f]

        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        x, w, b, conv_param = cache
        #dw, _ = Conv.forward(x, dout, 0, conv_param)

        # Récupération des variables
        x, w, b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        
        # Initialisations
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)
        
        # Dimensions
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, H_, W_ = dout.shape
        
        # db - dout (N, F, H', W')
        # On somme sur tous les éléments sauf les indices des filtres
        db = dout.sum((0, 2, 3))
        
        # dw = xp * dy
        # 0-padding juste sur les deux dernières dimensions de x
        xp = x.new_zeros((N, C, H + 2*pad, W + 2*pad))
        xp[:, :,pad:-pad,pad:-pad] = x
        #xp = np.pad(x, ((0,), (0,), (pad,), (pad, )), 'constant')
        
        # Version sans vectorisation
        for n in range(N):       # On parcourt toutes les images
            for f in range(F):   # On parcourt tous les filtres
                for i in range(HH): # indices du résultat
                    for j in range(WW):
                        for k in range(H_): # indices du filtre
                            for l in range(W_):
                                for c in range(C): # profondeur
                                    dw[f,c,i,j] += xp[n, c, stride*i+k, stride*j+l] * dout[n, f, k, l]

        # dx = dy_0 * w'
        # Valide seulement pour un stride = 1
        # 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
        doutp = dout.new_zeros((N, F, H_ + 2*(WW-1), W_ + 2*(HH-1)))
        doutp[:, :,(WW-1):-(WW-1),(HH-1):-(HH-1)] = dout

        #doutp = np.pad(dout, ((0,), (0,), (WW-1,), (HH-1, )), 'constant')
        
        # 0-padding juste sur les deux dernières dimensions de dx
        dxp = dx.new_zeros((N, C, H_ + 2*pad, W_ + 2*pad))
        dxp[:, :,pad:-pad,pad:-pad] = dx

        #dxp = np.pad(dx, ((0,), (0,), (pad,), (pad, )), 'constant')

        # filtre inversé dimension (F, C, HH, WW)
        w_ = torch.zeros_like(w)
        for i in range(HH):
            for j in range(WW):
                w_[:,:,i,j] = w[:,:,HH-i-1,WW-j-1]
        
        # Version sans vectorisation
        for n in range(N):       # On parcourt toutes les images
            for f in range(F):   # On parcourt tous les filtres
                for i in range(H+2*pad): # indices de l'entrée participant au résultat
                    for j in range(W+2*pad):
                        for k in range(HH): # indices du filtre
                            for l in range(WW):
                                for c in range(C): # profondeur
                                    dxp[n,c,i,j] += doutp[n, f, i+k, j+l] * w_[f, c, k, l]
        #Remove padding for dx
        dx = dxp[:,:,pad:-pad,pad:-pad]

        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        stride = pool_param['stride']
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']

        H_ = int(1 + (H - pool_height) / stride)
        W_ = int(1 + (W - pool_width) / stride)
        

        out = x.new_zeros((N, C, H_, W_))

        for c in range(C):
          for n in range(N):
            for h in range(H_):
              for w_ in range(W_):
                out[n,c,h,w_] = (x[n,c,h*stride:(h*stride+pool_height),w_*stride:(w_*stride+pool_width)]).max()

        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x, pool_param = cache
        #out, _ = MaxPool.forward(x, pool_param)
        N, C, H, W = x.shape
        stride = pool_param['stride']
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']

        H_ = int(1 + (H - pool_height) / stride)
        W_ = int(1 + (W - pool_width) / stride)
        dx = torch.zeros_like(x)
        for c in range(C):
          for n in range(N):
            for h in range(H_):
              for w_ in range(W_):
                dx[n,c,h*stride:(h*stride+pool_height),w_*stride:(w_*stride+pool_width)] = (x[n,c,h*stride:(h*stride+pool_height),w_*stride:(w_*stride+pool_width)] == x[n,c,h*stride:(h*stride+pool_height),w_*stride:(w_*stride+pool_width)].max()) * dout[n,c,h,w_]

        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        #torch.manual_seed(0)
        
        C, H, W = input_dims
        F = num_filters
        HH, WW = filter_size, filter_size
        
        (F, C, HH, WW)

        self.params["W1"] = weight_scale * torch.randn(
            F, C, HH, WW, dtype=dtype, device=device)

        self.params["W2"] = weight_scale * torch.randn(
            int(H * W * F / 4) , hidden_dim, dtype=dtype, device=device)

        self.params["W3"] = weight_scale * torch.randn(
            hidden_dim, num_classes, dtype=dtype, device=device)

        self.params["b1"] = torch.zeros(F, 
          dtype=dtype, device=device)

        self.params["b2"] = torch.zeros(hidden_dim, 
          dtype=dtype, device=device)

        self.params["b3"] = torch.zeros(num_classes, 
          dtype=dtype, device=device)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        N, C, H, W  = X.shape
        F = W1.shape[0]
        l_conv, cache_conv = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        l_hidden, cache_h = Linear_ReLU.forward(l_conv.reshape((N,-1)), W2, b2)
        #cache_H0, cache_H1 = cache_H
        scores, cache_scores = Linear.forward(l_hidden, W3, b3)
        if y == None:
          return scores
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        loss, dL = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1 * W1) + 
        torch.sum(W2 * W2) + torch.sum(W3 * W3))

        dH, dW3, db3 = Linear.backward(dL, cache_scores)
        dflatt, dW2, db2 = Linear_ReLU.backward(dH, cache_h)
        dX, dW1, db1 = Conv_ReLU_Pool.backward(dflatt.reshape((N,F,int(H/2),int(W/2))), cache_conv)

        grads['W1'] = dW1 + self.reg * 2 * W1
        grads['W2'] = dW2 + self.reg * 2 * W2
        grads['W3'] = dW3 + self.reg * 2 * W3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        C, H, W = input_dims
        F = num_filters
        self.num_filters = F
        HH, WW = 3, 3
        L = self.num_layers
        n_max_pool = len(max_pools)

        if weight_scale == 'kaiming':
          for i in range(L-1):
            if i == 0:
              Din = C
              Dout = F[i]
            else:
              Din = F[i-1]
              Dout = F[i]

            #self.params["W"+str(i+1)] = weight_scale * torch.randn(
            #  next_nfilt, prev_nfilt, HH, WW, dtype=dtype, device=device)
              
            self.params["W"+str(i+1)] = kaiming_initializer(Din, Dout, K=HH, relu=True, device=device,
                            dtype=dtype)

            self.params["b"+str(i+1)] = torch.zeros(Dout, 
            dtype=dtype, device=device)

            if batchnorm:
              self.params["gamma"+str(i+1)] = torch.ones(Dout, dtype=dtype, device=device)
              self.params["beta"+str(i+1)] = torch.zeros(Dout, dtype=dtype, device=device)

          #self.params["W"+str(L)] = weight_scale * torch.randn(
          #    int(H * W * F[L-2] / (2 ** (2*n_max_pool))) , num_classes, dtype=dtype, device=device)

          self.params["W"+str(L)] = kaiming_initializer(int(H * W * F[L-2] / (2 ** (2*n_max_pool))), num_classes, K=None, relu=True, device=device,
                            dtype=dtype)

          self.params["b"+str(L)] = torch.zeros(num_classes, 
            dtype=dtype, device=device)
        else:
          for i in range(L-1):
            if i == 0:
              prev_nfilt = C
              next_nfilt = F[i]
            else:
              prev_nfilt = F[i-1]
              next_nfilt = F[i]

            self.params["W"+str(i+1)] = weight_scale * torch.randn(
              next_nfilt, prev_nfilt, HH, WW, dtype=dtype, device=device)

            self.params["b"+str(i+1)] = torch.zeros(next_nfilt, 
            dtype=dtype, device=device)

            if batchnorm:
              self.params["gamma"+str(i+1)] = torch.ones(1, dtype=dtype, device=device)
              self.params["beta"+str(i+1)] = torch.zeros(1, dtype=dtype, device=device)

          self.params["W"+str(L)] = weight_scale * torch.randn(
              int(H * W * F[L-2] / (2 ** (2*n_max_pool))) , num_classes, dtype=dtype, device=device)

          self.params["b"+str(L)] = torch.zeros(num_classes, 
            dtype=dtype, device=device)

        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        batchnorm = self.batchnorm
        params = self.params
        bn_params = self.bn_params
        N = X.shape[0]
        L = self.num_layers
        layers = []
        caches = []
        max_pools = self.max_pools
        #if self.use_dropout:
        #    caches_drop = []

        for i in range(L - 1):
          Wi, bi = params["W"+str(i+1)], params["b"+str(i+1)]
          if batchnorm:
            gammai, betai = params["gamma"+str(i+1)], params["beta"+str(i+1)]
          if i == 0:
            if i in max_pools:
              if batchnorm:
                layer, cache = Conv_BatchNorm_ReLU_Pool.forward(X, Wi, bi, gammai, betai, conv_param, bn_params[i], pool_param)
              else:
                layer, cache = Conv_ReLU_Pool.forward(X, Wi, bi, conv_param, pool_param)
            else:
              if batchnorm:
                layer, cache = Conv_BatchNorm_ReLU.forward(X, Wi, bi, gammai, betai, conv_param, bn_params[i])
              else:
                layer, cache = Conv_ReLU.forward(X, Wi, bi, conv_param)
          else:    
            if i in max_pools:
              if batchnorm:
                layer, cache = Conv_BatchNorm_ReLU_Pool.forward(layers[i-1], Wi, bi, gammai, betai, conv_param, bn_params[i], pool_param)
              else:
                layer, cache = Conv_ReLU_Pool.forward(layers[i-1], Wi, bi, conv_param, pool_param)
            else:
              if batchnorm:
                layer, cache = Conv_BatchNorm_ReLU.forward(layers[i-1], Wi, bi, gammai, betai, conv_param, bn_params[i])
              else:
                layer, cache = Conv_ReLU.forward(layers[i-1], Wi, bi, conv_param)

          #if self.use_dropout:
          #  H, caches_d = Dropout.forward(H, self.dropout_param)
          #  caches_drop.append(caches_d)
          layers.append(layer)
          caches.append(cache)

        Wi, bi = params["W"+str(L)], params["b"+str(L)]
        scores, cache_scores = Linear.forward(layers[-1].reshape((N,-1)), Wi, bi)
        #if self.use_dropout:
        #    scores, caches_d = Dropout.forward(scores, self.dropout_param)
        #    caches_drop.append(caches_d)

        layers.append(scores)
        caches.append(cache_scores)

        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        loss, dL = softmax_loss(scores, y)
        dout = dL
        F = self.num_filters

        for i in range(L):
          loss += self.reg * (torch.sum(params["W"+str(i+1)] * params["W"+str(i+1)]))

          j = (L-1) - i

          #if self.use_dropout:
          #  cache_d = caches_drop.pop()
          #  dout = Dropout.backward(dout, cache_d)

          cache_out = caches.pop()
          if i == 0:
            dlayer, dWi, dbi = Linear.backward(dout, cache_out)
            aux_dim = int((dlayer.shape[1] / (F[-1])) ** 0.5)
            dlayer = dlayer.reshape((N,F[-1],aux_dim,aux_dim))
          else:
            if j in max_pools:
              if batchnorm:
                dlayer, dWi, dbi, dgammai, dbetai = Conv_BatchNorm_ReLU_Pool.backward(dout, cache_out)
              else:
                dlayer, dWi, dbi = Conv_ReLU_Pool.backward(dout, cache_out)
            else:
              if batchnorm:
                dlayer, dWi, dbi, dgammai, dbetai = Conv_BatchNorm_ReLU.backward(dout, cache_out)
              else:
                dlayer, dWi, dbi = Conv_ReLU.backward(dout, cache_out)

          dout = dlayer
          grads['W'+str(j+1)] = dWi + self.reg * 2 * params["W"+str(j+1)]
          grads['b'+str(j+1)] = dbi
          if batchnorm and i > 0:
            grads['gamma'+str(j+1)] = dgammai
            grads['beta'+str(j+1)] = dbetai

        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 1e-5   # Experiment with this!
    learning_rate = 1e-3  # Experiment with this!
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    model = DeepConvNet(input_dims=(3, 32, 32),
    num_classes=10,
    #num_filters=([8] * 2) + ([32] * 2)+ ([64] * 2) + ([128] * 2),
    num_filters=[8,16,32,64,128,256],
    max_pools=list(range(1,6)),
    weight_scale = 'kaiming',
    reg=1e-5,
    dtype=dtype, 
    device=device)

    solver = Solver(model, 
    data_dict, 
    update_rule=adam,
    optim_config={'learning_rate': 3e-3},
    print_every=100,
    num_epochs=15, 
    batch_size=128,
    lr_decay=0.99,
    verbose=False,
    device='cuda')
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        weight = torch.sqrt(torch.tensor(gain / Din)) * torch.randn(
            Din, Dout, dtype=dtype, device=device)
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        fan_in = Din * K * K
        weight = torch.sqrt(torch.tensor(gain / fan_in)) * torch.randn(
            Dout, Din, K, K,dtype=dtype, device=device)
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            sample_mean = x.mean(0)
            sample_var = x.var(0)
            out_ = (x - sample_mean) / (torch.sqrt(eps + sample_var))
            out = out_ * gamma + beta
            #print(x.shape)
            #print(running_mean.shape)
            #print(sample_mean.shape)
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            cache = (x, gamma, beta, bn_param, sample_mean, sample_var, out_)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            out = (x - running_mean) / (eps + torch.sqrt(running_var))
            out = out * gamma + beta
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        x, gamma, beta, bn_param, sample_mean, sample_var, out_ = cache
        eps = bn_param.get('eps', 1e-5)
        N, D = x.shape
        dx_n = dout * gamma
        dgamma = (dout * out_).sum(0)
        #dgamma = x.reshape((N, -1)).t().mm(dout) #+ reg * 2 * W2
        dbeta = dout.sum(0)

        sigma = torch.sqrt(sample_var + eps)
        #sigma_inv = 1/sigma[(sigma > 0)] * (sigma > 0)
        sigma_inv = 1 / sigma
        x_mu_0 = x - x.mean(0)

        #d_sigma_inv = x_mu_0.t().mm(dx_n)
        #d_sigma_inv = (x_mu_0 * dx_n).sum(0)
        #d_sigma = - d_sigma_inv * (sigma_inv * sigma_inv)
        #d_var = 0.5 * d_sigma * sigma_inv

        #dx_mu_0 = dx_n.mm(sigma_inv.t())
        #dx_mu_0 = dx_n.mm(torch.diag(sigma_inv).t())
        #dx_mu_0 = dx_n * sigma_inv
        #A = torch.eye(N, device=x.device) - x.new_ones((N,N)) / N
        #dx = A.mm(dx_mu_0) + 2 / N * A.mm(x * d_var)

          #unfold the variables stored in cache
        xmu = x - x.mean(0)

        #get the dimensions of the input/output
        

        #step9
        #dbeta = np.sum(dout, axis=0)
        dgammax = dout #not necessary, but more understandable

        #step8
        #dgamma = np.sum(dgammax*out_, axis=0)
        dout_ = dgammax * gamma

        #step7
        dsigma_inv = torch.sum(dout_*xmu, axis=0)
        dxmu1 = dout_ * sigma_inv

        #step6
        dsigma = -1. /(sigma**2) * dsigma_inv

        #step5
        dsample_var = 0.5 * 1. /torch.sqrt(sample_var+eps) * dsigma

        #step4
        dsq = 1. /N * x.new_ones((N,D)) * dsample_var

        #step3
        dxmu2 = 2 * xmu * dsq

        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * torch.sum(dxmu1+dxmu2, axis=0)

        #step1
        dx2 = 1. /N * x.new_ones((N,D)) * dmu

        #step0
        dx = dx1 + dx2

        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        out, cache = BatchNorm.forward(x.reshape((N, C,- 1)).reshape((-1,C)), 
        gamma, beta, bn_param)

        out = out.reshape((N, C,- 1)).reshape((N, C, H, W))

        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        N, C, H, W = dout.shape
        dx, dgamma, dbeta = BatchNorm.backward(dout.reshape((N, C,- 1)).reshape((-1,C)), cache)
        dx = dx.reshape((N, C,- 1)).reshape((N, C, H, W))
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
