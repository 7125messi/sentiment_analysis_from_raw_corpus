with open('./data/corpus_result01.txt',mode='r',encoding='utf-8') as fr:
    with open('./data/pos.csv',mode='w',encoding='utf-8') as f1:
        with open('./data/neutral.csv', mode='w', encoding='utf-8') as f2:
            with open('./data/neg.csv', mode='w', encoding='utf-8') as f3:
                for line in fr.readlines():
                    if int(line.strip()[-2:]) == 2:
                        f1.write(line.strip()[:-1] + '\n')
                    elif int(line.strip()[-1]) == 1:
                        f2.write(line.strip()[:-1] + '\n')
                    elif int(line.strip()[-2:]) == 0:
                        f3.write(line.strip()[:-1] + '\n')