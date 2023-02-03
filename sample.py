import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import json

def main(args):
    # Image preprocessing
    transform = transforms.Compose(
        [transforms.Resize(args.crop_size), transforms.CenterCrop(args.crop_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])# transform.Resize insteads of transforms.Scale()
        ##centecrop裁图片
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    # Build Models
    encoder = EncoderCNN(args.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)#必须使用，保证model中权值不变
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))  ##载入本地模型
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare Image
    """
     #read a single image, to transform it,and convert it to tensor
    image = Image.open(image)
    image_tensor = Variable(transform(image).unsqueeze(0))
    #read images under image_directory,do same operations 
    """
    image_dir =args.image_dir  ##测试文件夹地址
    images = os.listdir(image_dir)   ##将当前文件夹下的文件生成目录
    num_images = len(images)
    imgs=[]
   # imgs_tensor = torch.tensor()
    img_dict = {}
    for i, image in enumerate(images):
        img_path = os.path.join(image_dir, image)
       # img_dict[image] = str(img_path)
        with open(img_path, 'r+b') as f:
            with Image.open(f) as img:
                #resize image
                img =transform(img).unsqueeze(0)  ##在0位置插入一个1的维度
                img_dict[image] =img
               # imgs_tensor.cat((imgs_tensor,img),dim=0)#imgs.append(img)

    #image_tensor = Variable(transform(img).unsqueeze(0))
    # Set initial states
    state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
             Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))
        ###对tensor的封装？？？？？？？
    img_s ={}
    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()   ###加载到gpu上
        decoder.cuda()
        state = [s.cuda() for s in state]  ##加载到GPU
        #imgs_var = imgs_var.cuda()
        # Generate caption from image
        #for img in imgs:
        for img_key,img in img_dict.items():
            img_var = Variable(img)
            # print(img_var.shape)
            img_var = img_var.cuda()
            feature = encoder(img_var)

            sampled_ids = decoder.sample(feature, state)# """Generate captions for given image features using greedy search."""
            # print(sampled_ids.dtype)
            # print(sampled_ids.size())#(1,20)
            sampled_ids = sampled_ids.cpu().data.numpy()# convert words_id to numpy  tensor-》numpy
           ##print(sampled_ids.shape)  ##(1,20)
            sampled_ids = sampled_ids.flatten()#convert 2 d to 1d  按行降为降到1维
            ##print(sampled_ids.shape)   #(20)
            # Decode word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                   break

            sentence = ' '.join(sampled_caption)  ##拼接
            img_s[img_key] = sentence
            #Print out image and generated caption.
            #print(sentence)
            #plt.imshow(np.asarray(img[0][0]))
            '''
            plt.figure("Pytorch_Image_Caption")  # 图像窗口名称
            plt.imshow(img[0][0])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title("Image_Caption")  # 图像题目
            plt.text(0.5, 0.5, sentence, bbox=dict(facecolor='red', alpha=0.5))# plt.text(sentence)
            plt.show()
            '''
        #save the image and captions into the json form file
        json_dic = json.dumps(img_s, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=True)
        ##是把python对象转换成json对象生成一个fp的文件流，和文件相关
        print(json_dic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    parser.add_argument('--image', type=str,default='data2/image_datang_resized/image',
                        help='input image for generating caption')
    """
    parser.add_argument('--image_dir', type=str,default='data2/image_datang_resized/train2017',   ###注意此处！添加文件夹路径!
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-20-3000.ckpt',#default .pkl
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-20-3000.ckpt',#default.pkl
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data2/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=224, help='size for center cropping images')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    args=parser.parse_args()
    main(args)
    #args = parser.parse_args()
    #main(args)