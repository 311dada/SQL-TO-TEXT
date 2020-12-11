import argparse
from Utils.fix_seed import fix_seed
import torch
import logging
from Data.spider.dataset import SpiderDataset, SeqSpiderDataset
from Model.model import RGT, RelativeTransformer, AbsoluteTransformer, BiLSTM
from Utils.const import UP_SCHEMA_NUM, UP_TYPE_NUM, DOWN_TYPE_NUM, RGT_VOCAB_PATH, RGT_MODEL_PATH, SEQ_VOCAB_PATH, RELATIVE_MODEL_PATH, TRANSFORMER_MODEL_PATH, BILSTM_MODEL_PATH
import os
from torch.utils.data import DataLoader
from Data.spider.data_utils import get_RGT_batch_data, get_seq_batch_data
from Utils.metric import get_metric


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=str)
    parser.add_argument("--up_embed_dim", type=int)
    parser.add_argument("--down_embed_dim", type=int)
    parser.add_argument("--up_max_depth", type=int)
    parser.add_argument("--down_max_dist", type=int)
    parser.add_argument("--up_d_model", type=int)
    parser.add_argument("--down_d_model", type=int)
    parser.add_argument("--up_d_ff", type=int)
    parser.add_argument("--down_d_ff", type=int)
    parser.add_argument("--up_head_num", type=int)
    parser.add_argument("--down_head_num", type=int)
    parser.add_argument("--up_layer_num", type=int)
    parser.add_argument("--down_layer_num", type=int)
    parser.add_argument("--hid_size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--max_oov_num", type=int)
    parser.add_argument("--copy", type=int)
    parser.add_argument("--rel_share", type=int)
    parser.add_argument("--k_v_share", type=int)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--cross_atten", type=str)
    parser.add_argument("--up_rel", type=str, nargs='+')
    parser.add_argument("--down_rel", type=str, nargs='+')
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--train_step", type=int)
    parser.add_argument("--eval_step", type=int)
    parser.add_argument("--schedule_step", type=int)
    parser.add_argument("--log", type=str)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--output", type=str)

    return parser.parse_args()


def train(model, batch, label, optimizer, Loss, vocab_size, unk_idx, MODEL):
    out = None
    if MODEL == "RGT":
        up_x, up_type_x, down_x, down_type_x, up_depth, up_schema, down_dist, down_lca, q_x, AOA_mask, AOD_mask, copy_mask, src2trg_map = batch
        model.train()

        out = model(up_x, up_type_x, down_x, down_type_x, up_depth, up_schema,
                    down_dist, down_lca, q_x, AOA_mask, AOD_mask, copy_mask,
                    src2trg_map)
    elif MODEL in ["Relative-Transformer", "Transformer", "BiLSTM"]:
        nodes, questions, rela_dist, copy_mask, src2trg_map = batch

        if MODEL == "Relative-Transformer":
            out = model(nodes, rela_dist, questions, copy_mask, src2trg_map)
        elif MODEL in ["Transformer", "BiLSTM"]:
            out = model(nodes, questions, copy_mask, src2trg_map)
    else:
        # TODO
        pass

    out_dim = out.size(-1)
    out = out.reshape(-1, out_dim)
    trg = label.reshape(-1)

    out = torch.log(out + 1e-15)
    loss = Loss(out, trg)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    return loss.item()


def eval(model,
         dataset,
         up_vocab,
         down_vocab,
         args,
         device,
         MODEL,
         best_bleu=100,
         max_decode=50):
    model.eval()
    dataloader = DataLoader(dataset, args.eval_batch_size)

    total_preds = []

    for batch_data in dataloader:
        preds = []
        if MODEL == "RGT":
            batch, label = get_RGT_batch_data(batch_data, up_vocab.pad_idx,
                                              down_vocab.pad_idx, device,
                                              args.down_max_dist,
                                              down_vocab.size,
                                              down_vocab.unk_idx)
            up_x, up_type_x, down_x, down_type_x, up_depth, up_schema, down_dist, down_lca, q_x, AOA_mask, AOD_mask, copy_mask, src2trg_map = batch

            up_nodes, down_nodes, hidden, up_mask, down_mask = model.encode(
                up_x, up_type_x, down_x, down_type_x, up_depth, up_schema,
                down_dist, down_lca, AOA_mask, AOD_mask)

            inputs = q_x[:, 0].view(-1, 1)

            for i in range(max_decode):
                inputs = model.down_nodes_embed(inputs)
                cur_out, hidden = model.decode(inputs, up_nodes, down_nodes,
                                               hidden, up_mask, down_mask,
                                               copy_mask, src2trg_map)
                next_input = cur_out.argmax(dim=-1)
                preds.append(next_input)
                next_input[next_input >= down_vocab.size] = down_vocab.unk_idx
                inputs = next_input

        elif MODEL in ["Relative-Transformer", "Transformer", "BiLSTM"]:
            batch, label = get_seq_batch_data(batch_data, down_vocab.pad_idx,
                                              device, down_vocab.size,
                                              down_vocab.unk_idx,
                                              args.down_max_dist)
            nodes, questions, rela_dist, copy_mask, src2trg_map = batch

            if MODEL == "Relative-Transformer":
                nodes, hidden, mask = model.encode(nodes, rela_dist)
            elif MODEL in ["Transformer", "BiLSTM"]:
                nodes, hidden, mask = model.encode(nodes)
            else:
                # TODO
                pass

            inputs = questions[:, 0].view(-1, 1)

            for i in range(max_decode):
                cur_out, hidden = model.decode(inputs, nodes, hidden, mask,
                                               copy_mask, src2trg_map)
                next_input = cur_out.argmax(dim=-1)
                preds.append(next_input)
                next_input[next_input >= down_vocab.size] = down_vocab.unk_idx
                inputs = next_input

        else:
            # TODO
            pass
        preds = torch.cat(preds, dim=1)
        total_preds += preds.tolist()

    bleu, preds, refs = get_metric(total_preds, dataset.origin_questions,
                                   down_vocab, True, dataset.val_map_list,
                                   dataset.idx2tok_map_list)

    if bleu > best_bleu:
        with open(args.output, 'w') as f:
            for idx, pred in enumerate(preds):
                f.write(f"Pre: {pred}\n\n")
                f.write(f"Ref: \n     1. {refs[0][idx]}\n")
                if refs[0][idx] != refs[1][idx]:
                    f.write(f"     2. {refs[1][idx]}\n")
                f.write(f"{'-' * 60}\n")

    return bleu


def run(args):
    fix_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")

    up_vocab, down_vocab, vocab = None, None, None
    train_set, dev_set = None, None
    model = None

    # build vocabulary and load data
    if args.data == "spider":
        train_data_files = [
            "./Dataset/spider/train_spider.json",
            "./Dataset/spider/train_others.json"
        ]
        # train_data_files = ['./Dataset/spider/test.json']
        table_file = "./Dataset/spider/tables.json"
        dev_data_files = ["./Dataset/spider/dev.json"]

        if args.model == "RGT":
            train_set = SpiderDataset(train_data_files,
                                      table_file,
                                      min_freq=args.min_freq)
            dev_set = SpiderDataset(dev_data_files,
                                    table_file,
                                    down_vocab=train_set.down_vocab,
                                    up_vocab=train_set.up_vocab)
            up_vocab = train_set.up_vocab
            down_vocab = train_set.down_vocab

            if not os.path.exists(RGT_VOCAB_PATH):
                os.makedirs(RGT_VOCAB_PATH)
            up_vocab.save(os.path.join(RGT_VOCAB_PATH, 'up.vocab'))
            down_vocab.save(os.path.join(RGT_VOCAB_PATH, 'down.vocab'))
        elif args.model in ["Relative-Transformer", "Transformer", "BiLSTM"]:
            train_set = SeqSpiderDataset(train_data_files,
                                         table_file,
                                         min_freq=args.min_freq)
            dev_set = SeqSpiderDataset(dev_data_files,
                                       table_file,
                                       vocab=train_set.vocab)
            vocab = train_set.vocab

            if not os.path.exists(SEQ_VOCAB_PATH):
                os.makedirs(SEQ_VOCAB_PATH)
            vocab.save(os.path.join(SEQ_VOCAB_PATH, "relative.vocab"))

        else:
            # TODO
            pass

    elif args.data == "wikisql":
        # TODO
        pass
    else:
        raise NotImplementedError("Not supported dataset.")

    # build model
    if args.model == "RGT":
        model = RGT(args.up_embed_dim, args.down_embed_dim, up_vocab.size,
                    down_vocab.size, UP_TYPE_NUM, DOWN_TYPE_NUM, UP_SCHEMA_NUM,
                    args.up_max_depth, args.down_max_dist, args.up_d_model,
                    args.down_d_model, args.up_d_ff, args.down_d_ff,
                    args.up_head_num, args.down_head_num, args.up_layer_num,
                    args.down_layer_num, args.hid_size, args.dropout,
                    up_vocab.pad_idx, down_vocab.pad_idx, args.max_oov_num,
                    args.copy, args.rel_share, args.k_v_share, args.mode,
                    args.cross_atten, set(args.up_rel), set(args.down_rel))

    elif args.model == "Relative-Transformer":
        model = RelativeTransformer(args.down_embed_dim, vocab.size,
                                    args.down_d_model, args.down_d_ff,
                                    args.down_head_num, args.down_layer_num,
                                    args.hid_size, args.dropout, vocab.pad_idx,
                                    args.down_max_dist, args.max_oov_num,
                                    args.copy, args.rel_share, args.k_v_share)

    elif args.model == "Transformer":
        model = AbsoluteTransformer(args.down_embed_dim, vocab.size,
                                    args.down_d_model, args.down_d_ff,
                                    args.down_head_num, args.down_layer_num,
                                    args.hid_size, args.dropout, vocab.pad_idx,
                                    args.max_oov_num, args.copy)
    elif args.model == "BiLSTM":
        model = BiLSTM(args.down_embed_dim, vocab.size, args.hid_size,
                       vocab.pad_idx, args.dropout, args.max_oov_num,
                       args.copy)

    else:
        # TODO
        pass

    model.to(device)

    # optimizer
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.schedule_step,
                                                gamma=args.gamma)

    # loss function
    Loss = None
    if args.model == "RGT":
        Loss = torch.nn.NLLLoss(ignore_index=down_vocab.pad_idx)
    elif args.model in ["Relative-Transformer", "Transformer", "BiLSTM"]:
        Loss = torch.nn.NLLLoss(ignore_index=vocab.pad_idx)
    else:
        # TODO
        pass

    # data loader
    train_data_loader = DataLoader(train_set,
                                   batch_size=args.train_batch_size,
                                   shuffle=True)

    # if test_set is not None:
    #     test_data_loader = DataLoader(test_set,
    #                                   batch_size=args.eval_batch_size)

    # train and evaluate
    MODEL = None
    if args.model == "RGT":
        MODEL = RGT_MODEL_PATH
    elif args.model == "Relative-Transformer":
        MODEL = RELATIVE_MODEL_PATH
    elif args.model == "Transformer":
        MODEL = TRANSFORMER_MODEL_PATH
    elif args.model == "BiLSTM":
        MODEL = BILSTM_MODEL_PATH
    else:
        # TODO
        pass
    if not os.path.exists(MODEL):
        os.makedirs(MODEL)

    model_file = f"{args.prefix}.pt"
    model_path = os.path.join(MODEL, model_file)

    best_bleu = 0
    batch_step = 0

    for epoch in range(args.epoch):
        for batch_data in train_data_loader:
            batch, label = None, None
            if args.model == "RGT":
                batch, label = get_RGT_batch_data(batch_data, up_vocab.pad_idx,
                                                  down_vocab.pad_idx, device,
                                                  args.down_max_dist,
                                                  down_vocab.size,
                                                  down_vocab.unk_idx)
            elif args.model in ["Relative-Transformer", "Transformer", "BiLSTM"]:
                batch, label = get_seq_batch_data(batch_data, vocab.pad_idx,
                                                  device, vocab.size,
                                                  vocab.unk_idx,
                                                  args.down_max_dist)
            else:
                # TODO
                pass

            train_loss = 0
            if args.model == "RGT":
                train_loss = train(model, batch, label, optimizer, Loss,
                                   down_vocab.size, down_vocab.unk_idx,
                                   args.model)
            elif args.model in ["Relative-Transformer", "Transformer", "BiLSTM"]:
                train_loss = train(model, batch, label, optimizer, Loss,
                                   vocab.size, vocab.unk_idx, args.model)
            else:
                # TODO
                pass

            if batch_step and not batch_step % args.train_step:
                logging.info(
                    f"epoch {epoch}, batch {batch_step}: [training loss-> {round(train_loss, 3)}]"
                )

            if batch_step and not batch_step % args.eval_step:
                train_bleu, dev_bleu = 0, 0
                if args.model == "RGT":
                    train_bleu = eval(model, train_set, up_vocab, down_vocab,
                                      args, device, args.model)

                    logging.info(
                        f"epoch {epoch}, batch {batch_step}: [training bleu-> {round(train_bleu, 4)}]"
                    )
                    dev_bleu = eval(model,
                                    dev_set,
                                    up_vocab,
                                    down_vocab,
                                    args,
                                    device,
                                    args.model,
                                    best_bleu=best_bleu)
                    logging.info(
                        f"epoch {epoch}, batch {batch_step}: [dev bleu-> {round(dev_bleu, 4)}]"
                    )
                elif args.model in ["Relative-Transformer", "Transformer", "BiLSTM"]:
                    train_bleu = eval(model, train_set, None, vocab, args,
                                      device, args.model)

                    logging.info(
                        f"epoch {epoch}, batch {batch_step}: [training bleu-> {round(train_bleu, 4)}]"
                    )
                    dev_bleu = eval(model,
                                    dev_set,
                                    None,
                                    vocab,
                                    args,
                                    device,
                                    args.model,
                                    best_bleu=best_bleu)
                    logging.info(
                        f"epoch {epoch}, batch {batch_step}: [dev bleu-> {round(dev_bleu, 4)}]"
                    )
                else:
                    # TODO
                    pass

                if dev_bleu > best_bleu:
                    best_bleu = dev_bleu
                    torch.save({
                        "args": args,
                        "model": model.state_dict()
                    }, model_path)

            batch_step += 1

        scheduler.step()

    logging.info(f"best bleu: {round(best_bleu, 4)}")


if __name__ == "__main__":
    args = parse()
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename=args.log,
                        filemode='w')
    run(args)
