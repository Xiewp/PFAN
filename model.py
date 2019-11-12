import tensorflow as tf
import numpy as np
import math
def protoloss(sc,tc):
    return tf.reduce_mean(tf.square(sc-tc))

class AlexNetModel(object):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.featurelen=256
        self.source_moving_centroid=tf.get_variable(name='source_moving_centroid',shape=[num_classes,self.featurelen],initializer=tf.zeros_initializer(),trainable=False)
        self.target_moving_centroid=tf.get_variable(name='target_moving_centroid',shape=[num_classes,self.featurelen],initializer=tf.zeros_initializer(),trainable=False)

        tf.summary.histogram('source_moving_centroid',self.source_moving_centroid)
        tf.summary.histogram('target_moving_centroid',self.target_moving_centroid)



    def inference(self, x, training=False):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 1, 1e-5, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name ='pool2')
        norm2 = lrn(pool2, 1, 1e-5, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        
        conv4_flattened=tf.contrib.layers.flatten(conv4)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        self.flattened=flattened
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        if training:
            fc6 = dropout(fc6, self.dropout_keep_prob)
        self.fc6=fc6
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        if training:
            fc7 = dropout(fc7, self.dropout_keep_prob)
        self.fc7=fc7
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8=fc(fc7,4096,256,relu=False,name='fc8')
        self.vector=fc8
        self.fc8=fc8
        self.score = fc(fc8, 256, self.num_classes, relu=False, stddev=0.005,name='fc9')
        self.output=tf.nn.softmax(self.score/1.8)
        self.feature=self.fc8
        return self.score
    def adoptimize(self,learning_rate,train_layers=[]):
        var_list=[v for v in tf.trainable_variables() if 'D' in v.name]
        D_weights=[v for v in var_list if 'weights' in v.name]
        D_biases=[v for v in var_list if 'biases' in v.name]
        print '=================Discriminator_weights====================='
        print D_weights
        print '=================Discriminator_biases====================='
        print D_biases
        
        self.Dregloss=0.0005*tf.reduce_mean([tf.nn.l2_loss(v) for v in var_list if 'weights' in v.name])
        D_op1 = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(self.D_loss+self.Dregloss, var_list=D_weights)
        D_op2 = tf.train.MomentumOptimizer(learning_rate*2.0,0.9).minimize(self.D_loss+self.Dregloss, var_list=D_biases)
        D_op=tf.group(D_op1,D_op2)
        return D_op
    def wganloss(self,x,xt,batch_size,lam=10.0):
        with tf.variable_scope('reuse_inference') as scope:
            scope.reuse_variables()
            self.inference(x,training=True)
            source_fc6=self.fc6
            source_fc7=self.fc7
            source_fc8=self.fc8
            source_softmax=self.output
            source_output=outer(source_fc7,source_softmax)
            print 'SOURCE_OUTPUT: ',source_output.get_shape()
            scope.reuse_variables()
            self.inference(xt,training=True)
            target_fc6=self.fc6
            target_fc7=self.fc7
            target_fc8=self.fc8
            target_softmax=self.output
            target_output=outer(target_fc7,target_softmax)
            print 'TARGET_OUTPUT: ',target_output.get_shape()
        with tf.variable_scope('reuse') as scope:
            target_logits,_=D(target_fc8)
            scope.reuse_variables()
            source_logits,_=D(source_fc8)
            eps=tf.random_uniform([batch_size,1],minval=0.0,maxval=1.0)
            X_inter=eps*source_fc8+(1-eps)*target_fc8
            grad = tf.gradients(D(X_inter), [X_inter])[0]
            grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
            grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)
            D_loss=tf.reduce_mean(target_logits)-tf.reduce_mean(source_logits)+grad_pen
            G_loss=tf.reduce_mean(source_logits)-tf.reduce_mean(target_logits)	
            self.G_loss=G_loss
            self.D_loss=D_loss
            self.D_loss=0.3*self.D_loss
            self.G_loss=0.3*self.G_loss
            return G_loss,D_loss
    def adloss(self,x,xt,y,global_step):
        with tf.variable_scope('reuse_inference') as scope:
            scope.reuse_variables()
            self.inference(x,training=True)
            source_feature=self.feature
            scope.reuse_variables()
            self.inference(xt,training=True)
            target_feature=self.feature
            target_pred=self.output
        with tf.variable_scope('reuse') as scope:
            source_logits,_=D(source_feature)
            scope.reuse_variables()
            target_logits,_=D(target_feature)
            self.source_feature=source_feature
            self.target_feature=target_feature
            self.concat_feature=tf.concat([source_feature,target_feature],0)	
        source_result=tf.argmax(y,1)
        target_result=tf.argmax(target_pred,1)
        ones=tf.ones_like(source_feature)
        current_source_count=tf.unsorted_segment_sum(ones,source_result,self.num_classes)
        current_target_count=tf.unsorted_segment_sum(ones,target_result,self.num_classes)

        current_positive_source_count=tf.maximum(current_source_count,tf.ones_like(current_source_count))
        current_positive_target_count=tf.maximum(current_target_count,tf.ones_like(current_target_count))

        current_source_centroid=tf.divide(tf.unsorted_segment_sum(data=source_feature,segment_ids=source_result,num_segments=self.num_classes),current_positive_source_count)
        current_target_centroid=tf.divide(tf.unsorted_segment_sum(data=target_feature,segment_ids=target_result,num_segments=self.num_classes),current_positive_target_count)

        decay=tf.constant(0.3)
        self.decay=decay

#        target_centroid=(decay)*current_target_centroid+(1.-decay)*self.target_moving_centroid
        target_centroid=(decay)*current_target_centroid+(1.-decay)*self.source_moving_centroid
        source_centroid=(decay)*current_source_centroid+(1.-decay)*self.source_moving_centroid
	
        self.Semanticloss=protoloss(source_centroid,target_centroid)
        tf.summary.scalar('semanticloss',self.Semanticloss)

        D_real_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits,labels=tf.ones_like(target_logits)))
        D_fake_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=source_logits,labels=tf.zeros_like(source_logits)))
        self.D_loss=D_real_loss+D_fake_loss
        self.G_loss=-self.D_loss
        tf.summary.scalar('G_loss',self.G_loss)
        tf.summary.scalar('JSD',self.G_loss/2+math.log(2))
	
        self.G_loss=0.1*self.G_loss
        self.D_loss=0.1*self.D_loss
        return self.G_loss,self.D_loss,source_centroid,target_centroid

    def loss(self, batch_x, batch_y=None):
        with tf.variable_scope('reuse_inference') as scope:
            y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
        tf.summary.scalar('Closs',self.loss)
        return self.loss

    def optimize(self, learning_rate, train_layers,global_step,source_centroid,target_centroid):
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        print train_layers
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers+['fc9']]
        finetune_list=[v for v in var_list if v.name.split('/')[1] in ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']]
        new_list=[v for v in var_list if v.name.split('/')[1] in ['fc8','fc9']]
        self.Gregloss=0.0005*tf.reduce_mean([tf.nn.l2_loss(x) for x in var_list if 'weights' in x.name])
        finetune_weights=[v for v in finetune_list if 'weights' in v.name]
        finetune_biases=[v for v in finetune_list if 'biases' in v.name]
        new_weights=[v for v in new_list if 'weights' in v.name]
        new_biases=[v for v in new_list if 'biases' in v.name]
        print '==============finetune_weights======================='
        print finetune_weights
        print '==============finetune_biases======================='
        print finetune_biases
        print '==============new_weights======================='
        print new_weights
        print '==============new_biases======================='
        print new_biases
	
        self.F_loss=self.loss+self.Gregloss+global_step*self.G_loss+global_step*self.Semanticloss
        train_op1=tf.train.MomentumOptimizer(learning_rate*0.1,0.9).minimize(self.F_loss, var_list=finetune_weights)
        train_op2=tf.train.MomentumOptimizer(learning_rate*0.2,0.9).minimize(self.F_loss, var_list=finetune_biases)
        train_op3=tf.train.MomentumOptimizer(learning_rate*1.0,0.9).minimize(self.F_loss, var_list=new_weights)
        train_op4=tf.train.MomentumOptimizer(learning_rate*2.0,0.9).minimize(self.F_loss, var_list=new_biases)
        train_op=tf.group(train_op1,train_op2,train_op3,train_op4)
        with tf.control_dependencies([train_op1,train_op2,train_op3,train_op4]):
            update_sc=self.source_moving_centroid.assign(source_centroid)
            update_tc=self.target_moving_centroid.assign(target_centroid)
        return tf.group(update_sc,update_tc)
    def load_original_weights(self, session, skip_layers=[]):
        weights_dict = np.load('bvlc_alexnet.npy', encoding='bytes').item()
        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue
            if op_name == 'fc8' and self.num_classes != 1000:
                continue
            with tf.variable_scope('reuse_inference/'+op_name, reuse=True):
	        print '=============================OP_NAME  ========================================'
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases')
	        	print op_name,var
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights')
	        	print op_name,var
                        session.run(var.assign(data))


"""
Helper methods
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
        relu = tf.nn.relu(bias, name=scope.name)
        return relu
def D(x):
    with tf.variable_scope('D'):
        num_units_in=int(x.get_shape()[-1])
        num_units_out=1
        weights = tf.get_variable('weights',initializer=tf.truncated_normal([num_units_in,1024],stddev=0.01))
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.zeros_initializer())
        hx=(tf.matmul(x,weights)+biases)
        ax=tf.nn.dropout(tf.nn.relu(hx),0.5)
        weights2 = tf.get_variable('weights2',initializer=tf.truncated_normal([1024,1024],stddev=0.01))
        biases2 = tf.get_variable('biases2', shape=[1024], initializer=tf.zeros_initializer())
        hx2=(tf.matmul(ax,weights2)+biases2)
        ax2=tf.nn.dropout(tf.nn.relu(hx2),0.5)
        weights3 = tf.get_variable('weights3', initializer=tf.truncated_normal([1024,num_units_out],stddev=0.3))
        biases3 = tf.get_variable('biases3', shape=[num_units_out], initializer=tf.zeros_initializer())
        hx3=tf.matmul(ax2,weights3)+biases3
        return hx3,tf.nn.sigmoid(hx3)

def fc(x, num_in, num_out, name, relu=True,stddev=0.01):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', initializer=tf.truncated_normal([num_in,num_out],stddev=stddev))
        biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[num_out]))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def outer(a,b):
    a=tf.reshape(a,[-1,a.get_shape()[-1],1])
    b=tf.reshape(b,[-1,1,b.get_shape()[-1]])
    c=a*b
    return tf.contrib.layers.flatten(c)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1],
                          padding = padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
