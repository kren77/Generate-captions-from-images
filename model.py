import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):  ##编码器，使用CNN中的resnet152作为模型
    def __init__(self, embed_size):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)                                                    ##定义一个类，其中保存模型
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)                                ##线性传播输入维度为resnet的全连接层的输入维度，输出维度为embed_size
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)                                       ##对batch进行标准化

    def forward(self, images):  ##前向传播  输入为(batch_size,channel,images_size,image_size)
        """Extract feature vectors from input images."""
        with torch.no_grad():
             features = self.resnet(images)

        features = features.reshape(features.size(0), -1)                                                        ##转换成 features.size(0)行
        features = self.bn(self.linear(features))                                                                 ##线性网络的输出输入到batch中
        return features                                                                                          ##(batch_size,picture_size一般为14)


class DecoderRNN(nn.Module):  ##解码器
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)                                    ##词典共有vocab_size个词,每个词用embed_size个向量表示该词，为一个矩阵
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length                                                #最长句子长度

    def forward(self, features, captions, lengths):                                         ##输入features为encode的输出，captions为一个长度从大到小的的caption数组，lengths为对应captions
        #对应长度的列表也是从大到小排列
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)                                                         #embeddings'shape ==torch.size([128,23,256])   128,26,256

        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)                                                  #按维度1进行拼接，进行列进行拼接,why ?
                                                                                                  #features.unsqueeze(1) 's shape([128,1,256])
                                                                                                   #embeddings'shape ==torch.size([128,24,256])  128,27,256
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)                     ##对序列进行填充使得captions都相同长度，embeddings为pad的矩阵
        ##lengths为实际有用值的长度。
        hiddens, _ = self.lstm(packed)

        outputs = self.linear(hiddens[0])
        return outputs                                                                                                 ##[总共有多少个有用词汇,词汇表里有多少词汇]..hiddens,embeddings

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)##在1的位置插入一个1的维度
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size) ###sqyeeze()去掉维度位置1的维度
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length) 沿着位置1拼接
        return sampled_ids