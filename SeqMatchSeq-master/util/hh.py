filenames = {
            "train": '../data/' + 'wikiqa' + '/WikiQACorpus/WikiQA-train.txt',
            "dev": '../data/' + 'wikiqa' + '/WikiQACorpus/WikiQA-dev.txt',
            "test": '../data/' + 'wikiqa' + '/WikiQACorpus/WikiQA-test.txt'
        }
for folder, filename in filenames.items():
    data = []
    # Start a fresh instance, candidates, labels
    question, candidates, labels = [], [], []

    q = ''
    pos_num = 0

    for line in open(filename):
        divs = line.rstrip('\n').lower().split('\t')
        q_div, a_div, label = divs
        if q_div !=q:
            if pos_num >= 2:
                question.append(q)
                candidates.append(answers)
                labels.append(lab)
            answers = []
            answers.append(a_div)
            lab = []
            lab.append(label)
            pos_num = 0 + int(label)
            q = q_div
        else:
            answers.append(a_div)
            lab.append(label)
            if label == '1':
                pos_num += 1
    for i in range(len(question)):

        print(question[i])
        for j in range(len(candidates[i])):
            print(candidates[i][j])
            print(labels[i][j])

            print()
        print()
        print()




