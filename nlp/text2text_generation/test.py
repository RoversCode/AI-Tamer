# from transformers import AutoConfig, BertLayer


# config = AutoConfig.from_pretrained("bert-base-uncased")
# print(config)
# layer = BertLayer(config).half().cuda()
# print(layer)

import evaluate

if __name__ == '__main__':
    bleu = evaluate.load('bleu')
    