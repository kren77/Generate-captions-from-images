import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}   ##建立一个字典：与下面相反
        self.idx2word = {}   ##字典：{x: 'word',....}
        self.idx = 0          ##词数量

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx   ##添加词语
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:   ##如果词汇表里没有这些词语，那么输出['<unk>']
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)    ##返回词汇长度

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()#create the counter, default value is 0;
    ids = coco.anns.keys()

    # for i, id in enumerate(ids):   ##原版程序
    #     caption = str(coco.anns[id]['caption'])#get  the caption 'sentence' of each row of coco dataset
    #     tokens = nltk.tokenize.word_tokenize(caption.lower())#sentence tokenizer words for each captions: ['a', 'panoramic', 'view', 'of', 'a', 'kitchen', 'and', 'all', 'of', 'its', 'appliances', '.']
    #     counter.update(tokens)#add tokens to counter: Counter({'a': 1, 'very': 1, 'clean': 1, 'and': 1, 'well': 1, 'decorated': 1, 'empty': 1, 'bathroom': 1})
    #
    #     if (i+1) % 1000 == 0:
    #         print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))
    # print(ids)
    for i, id in enumerate(ids):
        # if i>=100:
        #     break
        # print(i,id)
        caption = str(coco.anns[id]['caption'])#get  the caption 'sentence' of each row of coco dataset
        tokens = nltk.tokenize.word_tokenize(caption.lower())#同时改为小写字母sentence tokenizer words for each captions: ['a', 'panoramic', 'view', 'of', 'a', 'kitchen', 'and', 'all', 'of', 'its', 'appliances', '.']
        counter.update(tokens)#add tokens to counter: Counter({'a': 41137, 'very': 1, 'clean': 1, 'and': 1, 'well': 1, 'decorated': 1, 'empty': 1, 'bathroom': 1})

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))
    # print (counter.items())
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]   ###counter.items返回可以遍历的值 word=词汇，cnt=数量
    # print (words)   #过滤小于频数的词汇
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()  ##创建一个类
    vocab.add_word('<pad>')   ##添加词汇
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    ## Add the words to the vocabulary.
    ## idx2word={0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>', 4: 'a', 5: 'very', 6: 'clean', 7: 'and',....}
    ## word2idx ={'<pad>':0,'<start>':1,...}

    # for i, word in enumerate(words):  ##原版
    #     vocab.add_word(word)
    # return vocab

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab
def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)   ##将对象封装到文件中
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
    # print(vocab.idx2word)
    # print(vocab.idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='D:/data/coco2017/annotations/captions_val2017.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='data2/vocab.pkl',  # './data/vocab.pkl'' is error
                        help='path for saving vocabulary wrapper')
    # parser.add_argument('--caption_path', type=str,
    #                     default= 'E:/datasets/COCO2014-2015/annotations/captions_train2014.json',
    #                     help='path for train annotation file')    ###原版
    # parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', #'./data/vocab.pkl'' is error
    #                     help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)