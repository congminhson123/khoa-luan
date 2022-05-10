from sklearn.model_selection import train_test_split

from modules.evaluate import cal_aspect_prf

from modules.aspect.dt_model import MebeAspectDTModel

from preprocess import load_aspect_data_mebe, preprocess

if __name__ == '__main__':
    inputs, outputs = load_aspect_data_mebe('data/raw_data/mebe_shopee.csv')
    inputs = preprocess(inputs)
    # corpus = []
    # vocab = []
    # with open('data/vocab/mebe_shopee_new_vocab.txt', encoding="utf8") as f:
    #     for line in f:
    #         vocab.append(line.rstrip("\n"))
    # # print(vocab)
    # # print(len(inputs))
    # for ip in inputs:
    #     # if ip.text in vocab:
    #     ipn = []
    #     iplist = list(ip.text.split(" "))
    #     for word in iplist:
    #         if word in vocab:
    #             ipn.append(word)
    #     ipn = " ".join(ipn)
    #     corpus.append(ipn)
    # # print(inputs[1].text[1])
    # vectorizer = TfidfVectorizer()
    # features = vectorizer.fit_transform(corpus)
    # # print(type(features))
    # inputs = features.toarray()
    # outputs = np.array(outputs)
    # # print(inputs.shape)
    # # print(len(outputs))
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=14)
    model = MebeAspectDTModel()
    # kf = KFold(n_splits=5, random_state=14, shuffle=False)
    # for train_index, test_index in kf.split(inputs, outputs):
    #     model.train(inputs[train_index], outputs[train_index])
    #     predicts = model.predict(inputs[test_index])
    #     print('\t\tship\t\t\t\tgiá\t\t\t\t\tchính hãng\t\t\t\tchất lượng\t\t\t\tdịch vụ\t\t\tan toàn\t\t\tothers')
    #     X = cal_aspect_prf(outputs[test_index], predicts, num_of_aspect=7, verbal=True)
    #     X
    model.train(X_train, y_train)

    predicts = model.predict(X_test)
    print('\t\tship\t\t\t\tgiá\t\t\t\t\tchính hãng\t\t\t\tchất lượng\t\t\t\tdịch vụ\t\t\tan toàn\t\t\tothers')
    X = cal_aspect_prf(y_test, predicts, num_of_aspect=7, verbal=True)
    print(type(X))
    X
