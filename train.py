import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN,DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([transforms.RandomCrop(args.crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) ##串联多个图片操作randomcrop（切割中心点位置）RHF（随机翻转）
    ##TOtensor(将其转换为[C,H,W],Normalize（正则化）

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True,
                             num_workers=args.num_workers)
    ##data_loader 里包含images,targets,lengths 三个tensor
    # for i,(images, captions, lengths) in enumerate(data_loader):
    #     print(lengths)
    #     print(captions)
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  ##损失函数
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    #将解码器的列表,编码器线性的列表，编码器的标准化（保持同分布）
    # print(params)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)      ##lr=学习率 params ##优化数据

    # Train the models
    total_step = len(data_loader)  ##结构体长度
    for epoch in range(args.num_epochs):   ##时间步
        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device) #shape = torch.size([128,23])
            # lengths = torch.tensor(lengths)
            # print(captions)
            # print(lengths)

            # print(captions.size()) [128,26]
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]


##pack_padded_sequence函数的参数，lengths需要从大到小排序，captions为已根据长度大小排好序，batch_first如果设置为true，则cap的第一维为batch_size，
# 第二维为seq_length，否则相反。

            #lengths =128,captions's shape()
            #Forward, backward and optimize
            features = encoder(images)#extract the image feature,shape(128,256)
            # print(features.shape)
            # # print(captions.shape)
            # # print(len(lengths))
            # print(t.shape)
            outputs = decoder(features, captions, lengths)

            # print(p.shape)
            # print(emb.shape)


            loss = criterion(outputs, targets)
            decoder.zero_grad() ##梯度初始化
            encoder.zero_grad()
            loss.backward()    ##反向传播
            optimizer.step()   ##根据梯度更新网络参数




            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.num_epochs, i,
                                                                                              total_step, loss.item(),
                                                                                              np.exp(loss.item())))
                ##输出每一步的信息

                # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data2/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='D:/专业课/上/数据科学开源工具/7_Pytorch_image_caption(NIC)/data2/image_datang_resized/val2017',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='D:/data/coco2017/annotations/captions_val2017.json',
                        help='path for train annotation json file')
    # parser.add_argument('--image_dir', type=str, default='E:/datasets/COCO2014-2015/images/resized/train2014', help='directory for resized images')
    # parser.add_argument('--caption_path', type=str, default='D:/data/coco2014/annotations/captions_train2014.json',
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=3000, help='step size for saving trained models')##1000

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)