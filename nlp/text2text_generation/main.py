#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2023/05/09 13:20:55
@Author  :   ChengHee 
@Version :   1.0
@Contact :   1059885524@qq.com
@Desc    :   None
"""

# here put the import lib
from functools import partial
import os
import sys
import random
import numpy as np
import transformers
import torch
import evaluate
from jieba import lcut
from torch import nn
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from utils.log import Logger
from utils.data_process import AdvertiseGen, batchify_fn
from utils.argugments import get_args

logger = Logger(__name__).get_logger()

# 设置随机种子
def set_env(seed: int = 42):
    from transformers import set_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    # if is_tf_available:
    #     tf.random.set_seed(seed)


def set_project_root(path):
    if path not in sys.path:
        os.chdir(path)
        # sys.path.insert(0, path)


def train(args):
    model_args, data_args, training_args = args
    # 令牌器
    tokenizer = MT5Tokenizer.from_pretrained(r"pretrained_models\nlp\mt5-base")
    # 获取数据
    collate_fn = partial(batchify_fn, args=data_args)
    logger.info("Loading training data...")
    train_data = AdvertiseGen(data_args.train_path, tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)
    logger.info(f"train data size: {len(train_data)}")
    logger.info(f"Loading eval data...")
    eval_data = AdvertiseGen(data_args.eval_path, tokenizer)
    eval_dataloader = DataLoader(eval_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    logger.info(f"eval data size: {len(eval_data)}")
    # 加载模型
    logger.info("Loading model...")
    model = MT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path).to(training_args.device)
    # 模型信息打印
    logger.info("Model information:")
    logger.info(model)
    # 模型参数打印
    logger.info("Model parameters:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.shape}")
    # 优化器
    logger.info("Loading optimizer...")
    optimizer = transformers.AdamW(model.parameters(), lr=training_args.learning_rate)
    # 模型训练
    logger.info("Start training...")
    per_epoch_steps = len(train_dataloader)
    bleu_score = 0 # BLEU score
    tolerance = 0 # 容忍度
    losses = 0
    model.train()
    for epoch in range(training_args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, target_ids, _ = batch
            # attention_mask = attention_mask.to(training_args.device)
            target_ids = target_ids.to(training_args.device)
            # target_attention_mask = target_attention_mask.to(training_args.device)
            output = model(input_ids=input_ids.to(training_args.device), 
                           attention_mask=attention_mask.to(training_args.device), 
                           labels=target_ids.to(training_args.device),
                           )
            # 计算损失
            # target_ids = nn.functional.one_hot(target_ids.view(-1), num_classes=output.logits.shape[-1])
            # target_ids = target_ids.float()
            # loss = loss_fn(output.logits.view(-1, output.logits.shape[-1]), target_ids)
            loss = output.loss
            losses += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % training_args.logging_steps == 0 and step != 0:
                logger.info(f"epoch:{epoch}  {step}/{per_epoch_steps} train_loss: {losses/training_args.logging_steps}")
                losses = 0
            if (step % training_args.eval_steps == 0 or step == per_epoch_steps) and step != 0:
                logger.info(f"do evaluation...")
                # 模型评估与保存
                bl = text_evaluate(training_args, eval_dataloader, model, tokenizer)
                if bl > bleu_score:
                    # 模型保存
                    logger.info("BLEU score is improved.")
                    bleu_score = bl
                    tolerance = 0
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, training_args.output_dir + f"/model_{epoch}_{step}.pth")
                else:
                    logger.info("BLEU score is not improved.")
                    tolerance += 1
                    if tolerance > 5: # 5次BLEU score没有提升，提前结束训练
                        logger.info("Early stop.")
                        return
                

def text_evaluate(args, data, model, tokenizer):
    """_summary_
    Args:
        args (_type_): _description_
        model (_type_): _description_
        tokenizer (_type_): _description_
    Return: BLEU score
    """
    targets = []
    preds = []
    model.eval()
    for batch in data:
        input_ids, _, _, batch_target = batch
        outputs = model.generate(input_ids.to(args.device),
                                 max_length=128,
                                 num_beams=5,
                                 length_penalty=1.2,
                                 temperature=0.8,
                                 num_return_sequences=1,
                                 eos_token_id=1,
                                 early_stopping=True
                    )
        # 解码
        batch_pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        targets.extend(batch_target)
        preds.extend(batch_pred)

    # 计算BLEU
    bleu_score = calc_bleu(preds, targets)
    
    # 预测保存
    with open(os.path.join(args.output_dir, "pred.txt"), "w", encoding="utf-8") as f:
        for pred, target in zip(preds,targets):
            f.write('预测:' + pred + "\t" + '标签:' + target + "\n")
            
    model.train()  # 恢复模型训练
    
    return bleu_score


def calc_bleu(preds, targets):
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )

    bleu = evaluate.load('bleu')
    # 分词 & 拼接
    preds = [" ".join(lcut(pred)) for pred in preds]
    targets = [" ".join(lcut(target)) for target in targets]
    # 计算BLEU
    bleu_score = bleu.compute(predictions=preds, references=targets)
    logger.info(f"BLEU : {bleu_score}")
    return bleu_score['bleu']


def main():
    # 设置程序运行根目录
    # set_project_root(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    print(sys.path[0])
    # 获取配置参数
    args = get_args()
    _, data_args, training_args = args
    # 设置随机种子
    set_env(training_args.seed)
    # 训练
    train(args)
    
    

if __name__ == "__main__":
    main()
    # with open("./data/AdvertiseGen/train.json", "r", encoding="utf-8") as f:
    #     pass
