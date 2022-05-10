import pandas as pd

if __name__ == '__main__':
    # f = open('data/vocab/mebe_shopee_vocab.txt','r', encoding="utf8")
    df = pd.read_csv('data/raw_data/mebe_shopee.csv', encoding="utf-8")
    vocab = list(set(' '.join([str(t).strip() for t in df['text']]).split()))
    # vocab = []
    # with open('data/vocab/mebe_shopee_vocab.txt', encoding="utf8") as f:
    #     for line in f:
    #         vocab.append(line.rstrip("\n"))
    # vocab = list(set(vocab))
    # vocab.sort()
    # with open('data/vocab/mebe_shopee_vocab.txt', 'w', encoding="utf-8") as f:
    with open('data/vocab/mebe_shopee_vocab.txt', 'w', encoding="utf-8") as f:
        for w in vocab:
            f.write('{}\n'.format(w))
    print(len(vocab))
    # vocab = []
    # for t in df['text']:
    #     try:
    #         vocab = list(set(' '.join([t.strip()]).split()))
    #     except:
    #         print(t)
    # vocab.sort()