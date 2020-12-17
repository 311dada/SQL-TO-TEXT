# predict with the optimal checkpoint
import argparse
import torch
from Data.dataset import RGTDataset, SeqDataset, SingleGraphDataset, TreeDataset
from Model.model import RGT, RelativeTransformer, AbsoluteTransformer, BiLSTM, GAT, GCN, TreeLSTM
from Utils.const import UP_SCHEMA_NUM, UP_TYPE_NUM, DOWN_TYPE_NUM, TYPE_NUM, RGT_VOCAB_PATH, RGT_MODEL_PATH, SEQ_VOCAB_PATH, RELATIVE_MODEL_PATH, TRANSFORMER_MODEL_PATH, BILSTM_MODEL_PATH, SINGLE_GRAPH_VOCAB_PATH, GAT_MODEL_PATH, GCN_MODEL_PATH, TREE_VOCAB_PATH, TREE_MODEL_PATH
import os
from torch.utils.data import DataLoader
from Data.utils import get_RGT_batch_data, get_seq_batch_data, get_single_graph_batch_data, get_tree_batch_data
from Utils.metric import get_metric


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch_size", type=int)

    return parser.parse_args()


def eval(model,
         dataset,
         up_vocab,
         down_vocab,
         args,
         device,
         MODEL,
         best_bleu=100,
         max_decode=50,
         write=False):
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

        elif MODEL in [
                "Relative-Transformer", "Transformer", "BiLSTM", "GAT", "GCN"
        ]:
            if MODEL in ["Relative-Transformer", "Transformer", "BiLSTM"]:
                batch, label = get_seq_batch_data(batch_data,
                                                  down_vocab.pad_idx, device,
                                                  down_vocab.size,
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
            elif MODEL in ["GAT", "GCN"]:
                batch, label = get_single_graph_batch_data(
                    batch_data, down_vocab.pad_idx, device, down_vocab.size,
                    down_vocab.unk_idx)

                nodes, types, questions, graphs, copy_mask, src2trg_map = batch
                nodes, hidden, mask = model.encode(nodes, types, graphs)

            inputs = questions[:, 0].view(-1, 1)

            for i in range(max_decode):
                cur_out, hidden = model.decode(inputs, nodes, hidden, mask,
                                               copy_mask, src2trg_map)
                next_input = cur_out.argmax(dim=-1)
                preds.append(next_input)
                next_input[next_input >= down_vocab.size] = down_vocab.unk_idx
                inputs = next_input

        else:
            batch, label = get_tree_batch_data(batch_data, device)

            nodes, types, node_order, adjacency_list, edge_order, questions, copy_mask, src2trg_map = batch

            nodes, hidden, mask = model.encode(nodes, types, node_order,
                                               adjacency_list, edge_order)

            inputs = questions[:, 0].view(-1, 1)

            for i in range(max_decode):
                cur_out, hidden = model.decode(inputs, nodes, hidden, mask,
                                               copy_mask, src2trg_map)
                next_input = cur_out.argmax(dim=-1)
                preds.append(next_input)
                next_input[next_input >= down_vocab.size] = down_vocab.unk_idx
                inputs = next_input

        preds = torch.cat(preds, dim=1)
        total_preds += preds.tolist()

    bleu, preds, refs = get_metric(total_preds, dataset.origin_questions,
                                   down_vocab, True, dataset.val_map_list,
                                   dataset.idx2tok_map_list)

    if write or bleu > best_bleu:
        with open(args.output, 'w') as f:
            for idx, pred in enumerate(preds):
                f.write(f"Pre: {pred}\n\n")
                f.write(f"Ref: {refs[idx]}\n")
                f.write(f"{'-' * 60}\n")

    return bleu


def run(Args):

    args = torch.load(Args.checkpoint)['args']
    device = torch.device(f"cuda:{Args.gpu}")
    args.data = Args.data
    args.eval_batch_size = Args.batch_size

    print(args.output)

    up_vocab, down_vocab, vocab = None, None, None
    train_set, dev_set = None, None
    model = None
    DATA = None
    train_data_files = []
    train_table_file = ''
    dev_data_files = []
    dev_table_file = ''

    # build vocabulary and load data
    if args.data == "spider":
        DATA = "spider"
        train_data_files = [
            "./Dataset/spider/train_spider.json",
            "./Dataset/spider/train_others.json"
        ]
        # train_data_files = ['./Dataset/spider/test.json']
        train_table_file = "./Dataset/spider/tables.json"
        dev_table_file = train_table_file
        dev_data_files = ["./Dataset/spider/dev.json"]
    elif args.data == "wikisql":
        DATA = "wikisql"
        train_data_files = ["./Dataset/wikisql/train.jsonl"]
        train_table_file = "./Dataset/wikisql/train.tables.jsonl"
        dev_data_files = ["./Dataset/wikisql/dev.jsonl"]
        dev_table_file = "./Dataset/wikisql/dev.tables.jsonl"
        test_data_files = ["./Dataset/wikisql/test.jsonl"]
        test_table_file = "./Dataset/wikisql/test.tables.jsonl"
    else:
        raise NotImplementedError("Not supported dataset.")

    if args.model == "RGT":
        train_set = RGTDataset(train_data_files,
                               train_table_file,
                               data=DATA,
                               min_freq=args.min_freq,
                               max_depth=args.up_max_depth)
        dev_set = RGTDataset(dev_data_files,
                             dev_table_file,
                             data=DATA,
                             down_vocab=train_set.down_vocab,
                             up_vocab=train_set.up_vocab,
                             max_depth=args.up_max_depth)

        if DATA == "wikisql":
            test_set = RGTDataset(test_data_files,
                                  test_table_file,
                                  data=DATA,
                                  down_vocab=train_set.down_vocab,
                                  up_vocab=train_set.up_vocab,
                                  max_depth=args.up_max_depth)
        up_vocab = train_set.up_vocab
        down_vocab = train_set.down_vocab

        rgt_vocab_path = os.path.join(RGT_VOCAB_PATH, DATA)
        if not os.path.exists(rgt_vocab_path):
            os.makedirs(rgt_vocab_path)
        up_vocab.save(os.path.join(rgt_vocab_path, "up.vocab"))
        down_vocab.save(os.path.join(rgt_vocab_path, "down.vocab"))
    elif args.model in ["Relative-Transformer", "Transformer", "BiLSTM"]:
        train_set = SeqDataset(train_data_files,
                               train_table_file,
                               data=DATA,
                               min_freq=args.min_freq)
        dev_set = SeqDataset(dev_data_files,
                             dev_table_file,
                             data=DATA,
                             vocab=train_set.vocab)
        if DATA == "wikisql":
            test_set = SeqDataset(test_data_files,
                                  test_table_file,
                                  data=DATA,
                                  vocab=train_set.vocab)
        vocab = train_set.vocab

        seq_vocab_path = os.path.join(SEQ_VOCAB_PATH, DATA)
        if not os.path.exists(seq_vocab_path):
            os.makedirs(seq_vocab_path)
        vocab.save(os.path.join(seq_vocab_path, "seq.vocab"))

    elif args.model in ["GAT", "GCN"]:
        train_set = SingleGraphDataset(train_data_files,
                                       train_table_file,
                                       data=DATA,
                                       min_freq=args.min_freq)
        dev_set = SingleGraphDataset(dev_data_files,
                                     dev_table_file,
                                     data=DATA,
                                     vocab=train_set.vocab)

        if DATA == "wikisql":
            test_set = SingleGraphDataset(test_data_files,
                                          test_table_file,
                                          data=DATA,
                                          vocab=train_set.vocab)
        vocab = train_set.vocab

        single_graph_vocab_path = os.path.join(SINGLE_GRAPH_VOCAB_PATH, DATA)
        if not os.path.exists(single_graph_vocab_path):
            os.makedirs(single_graph_vocab_path)
        vocab.save(os.path.join(single_graph_vocab_path, "SingleGraph.vocab"))

    elif args.model == "TreeLSTM":
        train_set = TreeDataset(train_data_files,
                                train_table_file,
                                data=DATA,
                                min_freq=args.min_freq)
        dev_set = TreeDataset(dev_data_files,
                              dev_table_file,
                              data=DATA,
                              vocab=train_set.vocab)

        if DATA == "wikisql":
            test_set = TreeDataset(test_data_files,
                                   test_table_file,
                                   data=DATA,
                                   vocab=train_set.vocab)
        vocab = train_set.vocab

        tree_vocab_path = os.path.join(TREE_VOCAB_PATH, DATA)
        if not os.path.exists(tree_vocab_path):
            os.makedirs(tree_vocab_path)
        vocab.save(os.path.join(tree_vocab_path, "tree.vocab"))

    else:
        raise ValueError("Not supported model.")

    # build model
    if args.model == "RGT":
        model = RGT(args.up_embed_dim, args.down_embed_dim, up_vocab.size,
                    down_vocab.size, UP_TYPE_NUM, DOWN_TYPE_NUM, UP_SCHEMA_NUM,
                    args.up_max_depth, args.down_max_dist, args.up_d_model,
                    args.down_d_model, args.up_d_ff, args.down_d_ff,
                    args.up_head_num, args.down_head_num, args.up_layer_num,
                    args.down_layer_num, args.hid_size, args.dropout,
                    up_vocab.pad_idx, down_vocab.pad_idx, args.max_oov_num,
                    args.copy, args.rel_share, args.k_v_share,
                    args.mode, args.cross_atten, set(args.up_rel),
                    set(args.down_rel), DATA)

    elif args.model == "Relative-Transformer":
        model = RelativeTransformer(args.down_embed_dim, vocab.size,
                                    args.down_d_model, args.down_d_ff,
                                    args.down_head_num, args.down_layer_num,
                                    args.hid_size, args.dropout, vocab.pad_idx,
                                    args.down_max_dist, args.max_oov_num,
                                    args.copy, args.rel_share, args.k_v_share)

    elif args.model == "Transformer":
        model = AbsoluteTransformer(args.down_embed_dim,
                                    vocab.size,
                                    args.down_d_model,
                                    args.down_d_ff,
                                    args.down_head_num,
                                    args.down_layer_num,
                                    args.hid_size,
                                    args.dropout,
                                    vocab.pad_idx,
                                    max_oov_num=args.max_oov_num,
                                    copy=args.copy,
                                    pos=args.absolute_pos)
    elif args.model == "BiLSTM":
        model = BiLSTM(args.down_embed_dim, vocab.size, args.hid_size,
                       vocab.pad_idx, args.dropout, args.max_oov_num,
                       args.copy)

    elif args.model == "GAT":
        model = GAT(args.down_embed_dim, TYPE_NUM, vocab.size,
                    args.down_d_model, args.down_d_ff, args.down_head_num,
                    args.down_layer_num, args.hid_size, vocab.pad_idx,
                    args.dropout, args.max_oov_num, args.copy)

    elif args.model == "GCN":
        model = GCN(args.down_embed_dim, vocab.size, TYPE_NUM, args.hid_size,
                    args.down_layer_num, vocab.pad_idx, args.dropout,
                    args.max_oov_num, args.copy)
    else:
        model = TreeLSTM(args.down_embed_dim, vocab.size, TYPE_NUM,
                         args.hid_size, args.dropout, vocab.pad_idx,
                         args.max_oov_num, args.copy)
        args.train_batch_size = 1
        args.eval_batch_size = 1

    model.to(device)
    model.load_state_dict(torch.load(Args.checkpoint)['model'])

    if DATA == "spider":
        test_set = dev_set

    if args.model == "RGT":
        bleu = eval(model,
                    test_set,
                    up_vocab,
                    down_vocab,
                    args,
                    device,
                    args.model,
                    write=True)
    elif args.model in [
            "Relative-Transformer", "Transformer", "BiLSTM", "GAT", "GCN",
            "TreeLSTM"
    ]:
        bleu = eval(model,
                    test_set,
                    None,
                    vocab,
                    args,
                    device,
                    args.model,
                    write=True)
    else:
        raise ValueError("Not supported model.")

    print(f"bleu: {round(bleu, 4)}")


if __name__ == "__main__":
    args = parse()

    run(args)
