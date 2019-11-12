import os
import numpy as np
import tensorflow as tf
import datetime
from model import AlexNetModel
from dataprocessor import BatchPreprocessor
import math
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 10000000, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size',100, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '256,257', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')

NUM_CLASSES = 31
TRAINING_FILE = 'amazon_list.txt'
VAL_FILE = 'webcam_list.txt'
FLAGS = tf.app.flags.FLAGS
MAX_STEP=10000
MODEL_NAME='amazo_to_webcam_final'
def decay(start_rate,epoch,num_epochs):
    return start_rate/pow(1+0.001*epoch,0.75)

def adaptation_factor(x):
	if x>=1.0:
		return 1.0
	den=1.0+math.exp(-10*x)
	lamb=2.0/den-1.0
	return lamb

def cos_distance(vector1,vector2):  
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for a,b in zip(vector1,vector2):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return None  
    else:  
        return dot_product / ((normA*normB)**0.5)


def main(_):
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()
    # Placeholders
    x = tf.placeholder(tf.float32, [None, 227, 227, 3],'x')
    xt = tf.placeholder(tf.float32, [None, 227, 227, 3],'xt')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES],'y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES],'yt')
    adlamb=tf.placeholder(tf.float32)
    decay_learning_rate=tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel(num_classes=NUM_CLASSES, dropout_keep_prob=dropout_keep_prob)
    loss = model.loss(x, y)
    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    correct=tf.reduce_sum(tf.cast(correct_pred,tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    G_loss,D_loss,sc,tc=model.adloss(x,xt,y,adlamb)
    target_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    target_correct=tf.reduce_sum(tf.cast(target_correct_pred,tf.float32))
    target_accuracy = tf.reduce_mean(tf.cast(target_correct_pred, tf.float32))
    train_op = model.optimize(decay_learning_rate, train_layers,adlamb,sc,tc)
    
    
    # Testing accuracy of the model
    source_vector = model.fc8
    target_vector = model.vector
    target_pre = tf.argmax(model.score, 1)
    
    
    
	
    D_op=model.adoptimize(decay_learning_rate,train_layers)
    optimizer=tf.group(train_op,D_op)
    
    
    train_writer=tf.summary.FileWriter('./log/tensorboard'+MODEL_NAME)
    train_writer.add_graph(tf.get_default_graph())
    tf.summary.scalar('Testing Accuracy',target_accuracy)
    merged=tf.summary.merge_all()

    print '============================GLOBAL TRAINABLE VARIABLES ============================'
    print tf.trainable_variables()
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2: 
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None
    print '==================== MULTI SCALE==================================================='
    print multi_scale
    train_preprocessor = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                           output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    Ttrain_preprocessor = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                           output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES, output_size=[227, 227],multi_scale=multi_scale,istraining=False)
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    Ttrain_batches_per_epoch = np.floor(len(Ttrain_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    
    
    dic_s = {}
    dic_temp = {}
    dic_temp1 = {}
    dic_t = {}
    


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        train_writer.add_graph(sess.graph)
        model.load_original_weights(sess, skip_layers=train_layers)
        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))
        gs=0
        gd=0
        best_acc = 0.0
        flag = 1
        flag1 = 1
        first_s = 50
        for epoch in range(FLAGS.num_epochs):
            step = 1
            while step < train_batches_per_epoch:
                gd+=1
                lamb=adaptation_factor(gd*1.0/MAX_STEP)
                rate=decay(FLAGS.learning_rate,gd,MAX_STEP)
                for it in xrange(1):
                    gs+=1
                    if gs%Ttrain_batches_per_epoch==0:
                        Ttrain_preprocessor.reset_pointer()
                    if gs%train_batches_per_epoch==0:
                        train_preprocessor.reset_pointer()
                    batch_xs, batch_ys = train_preprocessor.next_batch(FLAGS.batch_size)
                    Tbatch_xs, Tbatch_ys = Ttrain_preprocessor.next_batch(FLAGS.batch_size)
                    summary,_=sess.run([merged,optimizer], feed_dict={x: batch_xs,xt: Tbatch_xs,yt:Tbatch_ys,adlamb:lamb, decay_learning_rate:rate,y: batch_ys,dropout_keep_prob:0.5})
                    train_writer.add_summary(summary,gd)
                    closs,gloss,dloss,gregloss,dregloss,floss,smloss=sess.run([model.loss,model.G_loss,model.D_loss,model.Gregloss,model.Dregloss,model.F_loss,model.Semanticloss],
                                                                              feed_dict={x: batch_xs,xt: Tbatch_xs,adlamb:lamb, decay_learning_rate:rate,y: batch_ys,dropout_keep_prob:0.5})
                step += 1
                
                
                if epoch == first_s:
                    source_v = sess.run(source_vector, feed_dict={x: batch_xs, y: batch_ys, xt: Tbatch_xs,dropout_keep_prob: 1.})
                    for i in range(FLAGS.batch_size):
                        dic_temp.setdefault(np.argmax(batch_ys[i]),[]).append(source_v[i])
                if epoch == first_s+1 and flag == 1:
                    for i in dic_temp.keys():
                        dic_s[i] = np.mean(dic_temp[i],axis=0)
                    with open('dic_s.txt','w') as f:
                        f.write(str(dic_s))
                    flag = 0
                
                
                
#                if gd%50==0:
                if epoch%5 == 0 and step == train_batches_per_epoch - 1:
                    print '=================== Step {0:<10} ================='.format(gs)
                    print 'Epoch {0:<5} Step {1:<5} Closs {2:<10} Gloss {3:<10} Dloss {4:<10} Total_Loss {7:<10} Gregloss {5:<10} Dregloss {6:<10} Semloss {7:<10}'.format(epoch,step,closs,gloss,dloss,gregloss,dregloss,floss,smloss)
                    print 'lambda: ',lamb
                    print 'rate: ',rate
                    # Epoch completed, start validation
                    print("{} Start validation".format(datetime.datetime.now()))
                    test_acc = 0.
                    test_count = 0
                    fp = open('pre_and_sim.txt','w')
                    for _ in range((len(val_preprocessor.labels))):
                        batch_tx, batch_ty = val_preprocessor.next_batch(1)
                        
                        if flag == 0 and flag1 == 1:
                            target_v = sess.run(target_vector, feed_dict={xt: batch_tx, dropout_keep_prob: 1.})
                            sim_list = []
                            for j in range(NUM_CLASSES):
#                                print(target_v[0])
#                                print('okkk')
#                                print(dic_s[j])
                                sim_value = cos_distance(target_v[0], dic_s[j])
                                sim_list.append(sim_value)
                            max_sim = max(sim_list)
                            max_idx = sim_list.index(max_sim)
                            fp.write(str(max_idx) + ' ' + str(max_sim) + '\n')
                            dic_temp1.setdefault(np.argmax(batch_ty[0]),[]).append(target_v[0])
                        if epoch > first_s and flag1 == 1:
                            for i in dic_temp1.keys():
                                dic_t[i] = np.mean(dic_temp1[i],axis=0)
                            with open('dic_t.txt','w') as f:
                                f.write(str(dic_t))
                            flag1 = 0
                        
                        
                        acc = sess.run(correct, feed_dict={x: batch_tx, y: batch_ty, dropout_keep_prob: 1.})
                        test_acc += acc
                        test_count += 1
                    fp.close()
                    print test_acc,test_count
                    test_acc /= test_count
                    if test_acc > best_acc:
                        best_acc = test_acc
                    print('best acc is: %f'%best_acc)
                    print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
                    # Reset the dataset pointers
                    val_preprocessor.reset_pointer()
                    #train_preprocessor.reset_pointer()
                if gd%4000==0 and gd>0:
                    saver.save(sess,'./log/mstnmodel_'+MODEL_NAME+str(gd)+'.ckpt')
                    print("{} Saving checkpoint of model...".format(datetime.datetime.now()))
#                    while(1):
#                        print("1")


if __name__ == '__main__':
    tf.app.run()
