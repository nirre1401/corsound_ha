from __future__ import division
from __future__ import print_function
from sklearn.utils import resample
import argparse
import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import pandas as pd
from numpy import random
from sklearn import preprocessing
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import pickle
from retrieval_model import FOP

os.environ['CUDA_VISIBLE_DEVICES'] = "0"




def get_batch(batch_index, batch_size, labels, f_lst):
    '''
    prepares a batch of data for training
    '''
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])
 
def init_weights(m):
    '''
    init NN weights
    '''
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main(train_data, test_data, train_label, test_label):
    '''
    this is the main flow of the training:
    1. It builds and compile the model
    2. Init weights by Xavier method https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    3. init network hyper-params
    4. Train by bathes and output accuracy and loss
    5. returns a trained model
    '''

    n_class = 2

    test_data = torch.from_numpy(test_data).float()
    model = FOP(FLAGS, train_data.shape[1], n_class)
    model.apply(init_weights)

    ce_loss = nn.CrossEntropyLoss()
    opl_loss = OrthogonalProjectionLoss()

    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()
        opl_loss.cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))


    for alpha in FLAGS.alpha_list:
        epoch=1
        num_of_batches = (len(train_label) // FLAGS.batch_size)
        loss_plot = []
        loss_per_epoch = 0
        s_fac_per_epoch = 0
        d_fac_per_epoch = 0
        txt_dir = 'output'
        save_dir = 'fc2_%s_%s_alpha_%0.2f'%(FLAGS.split_type, FLAGS.save_dir, alpha)
        txt = '%s/ce_opl_%03d_%0.2f.txt'%(txt_dir, FLAGS.max_num_epoch, alpha)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        with open(txt,'w+') as f:
            f.write('EPOCH\tLOSS\tEER\tAUC\tS_FAC\tD_FAC\n')

        save_best = 'best_%s'%(save_dir)

        if not os.path.exists(save_best):
            os.mkdir(save_best)
        with open(txt,'a+') as f:
            while (epoch < FLAGS.max_num_epoch):
                print('%s\tEpoch %03d'%(FLAGS.split_type, epoch))
                for idx in tqdm(range(num_of_batches)):
                    train_batch, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, train_data)
                    # voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
                    loss_tmp, loss_opl, loss_soft, s_fac, d_fac = train(train_batch,
                                                                 batch_labels,
                                                                 model, optimizer, ce_loss, opl_loss, alpha)
                    loss_per_epoch+=loss_tmp
                    s_fac_per_epoch+=s_fac
                    d_fac_per_epoch+=d_fac

                loss_per_epoch/=num_of_batches
                s_fac_per_epoch/=num_of_batches
                d_fac_per_epoch/=num_of_batches
                # predict

                output = model(test_data)

                loss_plot.append(loss_per_epoch)
                output = [(o[1] > o[0]).cpu().detach().numpy() for o in output[1]]
                correct = output == test_label
                accuracy = np.sum(correct) / len(test_label)
                print("Epoch {} | Loss: {:.3f}, Accuracy: {:.3f}".format(epoch, loss_per_epoch, accuracy))

                # save_checkpoint({
                #     'epoch': epoch,
                #     'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d_%0.3f.pth.tar'%(epoch, accuracy))

                print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f, Accuracy: %0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch, alpha, accuracy))

    return model


def predict(model, image_embed, voice_embed):
    '''
    for external validation or the expose predict for prod uses
    '''
    data = np.column_stack((image_embed, voice_embed))
    data = np.asarray(data).astype(float)
    data_flatten = np.asarray(data).flatten()
    output = model(data_flatten)
    output = [(o[1] > o[0]).cpu().detach().numpy() for o in output[1]]
    return output

def read_embeddings(image_path = 'data/vfm_assignment/image_embeddings.pickle', voice_path = 'data/vfm_assignment/audio_embeddings.pickle'):
    '''
    Load the embedding from a serialized pickle of face images and voice spectograms
    '''
    os.chdir('..')
    images_embedding_path = image_path
    images_embeddings = pickle.load(open(images_embedding_path, 'rb'))
    img_emb_labels = [[k.split("/")[0],v] for k,v in images_embeddings.items()]

    voice_embedding_path = voice_path
    voice_embeddings = pickle.load(open(voice_embedding_path, 'rb'))
    voic_emb_labels = [[k.split("/")[0], v] for k, v in voice_embeddings.items()]

    return img_emb_labels, voic_emb_labels


def read_data(FLAGS):
    '''
    1. load embedding of faces and voices
    2. Create a training set by cartesian product of all faces and voice and placing match = 1 where the label (person name) of the face voice match
    3. Balance the training set (because the cartesian product it is unbalanced towards 0 (unmatched face-voice pair)
    4. Partition randomly to train and test sets
    5. Return datasets for training
    '''
    images_emb_labels, voice_emb_labels = read_embeddings()
    face_train_df = pd.DataFrame(images_emb_labels, columns=['label', 'image_data'])
    voice_train_df = pd.DataFrame(images_emb_labels, columns=['label', 'voice_data'])
    # merge voices and faces and place 1 where matched
    match_df = pd.merge(face_train_df.assign(key=1), voice_train_df.assign(key=1), on='key').drop('key', axis=1)
    match_df['label'] = np.where(match_df['label_x'] == match_df['label_y'], 1, 0)

    df_majority = match_df[match_df.label == 0]
    df_minority = match_df[match_df.label == 1]
    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=df_minority.shape[0])
    df_merged = [df_majority_downsampled, df_minority]
    match_df = pd.concat(df_merged)
    print('Split Type: %s'%(FLAGS.split_type))

    if FLAGS.split_type == 'random':
        arr_faces = match_df.image_data.to_numpy()
        arr_faces = np.stack(arr_faces)
        arr_voices = match_df.voice_data.to_numpy()
        arr_voices = np.stack(arr_voices)
        train_data = np.column_stack((arr_faces, arr_voices))
        train_label = np.vstack((match_df.label))
        combined = list(zip(train_data, train_label))
        random.shuffle(combined)
        train_data, train_label = zip(*combined)
        train_data = np.asarray(train_data).astype(float)
        train_label = np.asarray(train_label).flatten()


    # SPLIT to TRAIN and TEST
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size = 0.33, random_state = 42)

    print("Train file length", len(train_data))
    print('Shuffling\n')

    return X_train, X_test, y_train, y_test

class OrthogonalProjectionLoss(nn.Module):
    '''
    https://arxiv.org/pdf/2103.14021.pdf
    '''
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels=None):
        '''
        feed forward data into the NN
        '''
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


def train(train_batch, labels, model, optimizer, ce_loss, opl_loss, alpha):
    '''
    train a mini-batch by feed forward, calculaing the loss , backpropogate the loss and updating the NN weights accordingly
    '''
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    train_batch = torch.from_numpy(train_batch).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        train_batch, labels = train_batch.cuda(), labels.cuda()

    train_batch, labels = Variable(train_batch), Variable(labels)
    comb = model.train_forward(train_batch)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
    loss_soft = ce_loss(comb[1], labels)
    
    loss = loss_soft + alpha * loss_opl

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

class RunningAverage(object):
    '''
    calculate average as we go and accumulate more and more loss data points
    '''
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average
 
def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='CUDA Training')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory for saving checkpoints.')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-4)') 
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--max_num_epoch', type=int, default=10, help='Max number of epochs to train, number')
    parser.add_argument('--alpha_list', type=list, default=[1], help='Alpha Values List')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--split_type', type=str, default='random', help='split_type')

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    X_train, X_test, y_train, y_test = read_data(FLAGS)

    
    model = main(X_train, X_test, y_train, y_test)

    # If you want to test the model with new data use this call
    # 1. Please for each example
    #embeddings_img, embeddings_voice  = read_embeddings(image_path, voice_path)
    #external_validation_results = predict(embeddings_img, embeddings_voice)


