#!/usr/bin/env python3
# encoding: utf-8
# /nfs_beijing/kubeflow-user/jinxian_2024/dynamicbind/bin/python /nfs_beijing_ai/jinxian/GNNRefine/code/GNNRefine.py /nfs_beijing_ai/jinxian/GNNRefine/data/pred_path/pred_path/casp15_af_m1.txt /nfs_beijing_ai/jinxian/GNNRefine/data/output/casp15_af_m1_Refine -n_proc=12
# /nfs_beijing/kubeflow-user/jinxian_2024/dynamicbind/bin/python /nfs_beijing_ai/jinxian/GNNRefine/code/GNNRefine.py /nfs_beijing_ai/jinxian/GNNRefine/data/pred_path/pred_path/casp15_af3.txt /nfs_beijing_ai/jinxian/GNNRefine/data/output/casp15_af3_Refine -n_proc=12
from __future__ import print_function, division
import os, pickle, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import argparse

from Data import Data
from Model import GNN

__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]

class GNNPred:
    def __init__(self, device_id, distCB=True, QA=False):
        """
        Args:
            device_id (int): the GPU id, -1 means cpu.
            distCB: GNN model for refine distance prediction.
            QA: GNN model for quality assessment task.
        """
        self.distCB, self.QA = distCB, QA
        assert (not distCB==QA), "please select only one task: distCB or QA."
        self.device = torch.device('cuda:%s'%(device_id)) if device_id>=0 else torch.device('cpu')
        self.model = GNN(distCB=distCB, QA=QA).to(self.device)

    def load_model(self, params_file):
        print('Load model params.')
        model_dict = torch.load(params_file, map_location=self.device)
        self.model.load_state_dict(model_dict)

    def forward(self, sample):
        # build graph
        node_feat, edge_feat, adj = sample['feature']['node'].squeeze(0), sample['feature']['edge'].squeeze(0), sample['feature']['adj'].squeeze(0)
        graph = dgl.DGLGraph()
        graph.add_nodes(node_feat.shape[0])
        ii, jj = np.where(adj==1)
        graph.add_edges(ii, jj)
        graph.ndata['nfeat'] = node_feat
        graph.edata['efeat'] = edge_feat[ii,jj]

        # atom_emb
        atom_emb = sample['feature']['atom_emb']
        graph.ndata['atom_emb'] = atom_emb['embedding'].squeeze(0)

        # forward
        output = self.model(graph.to(self.device))

        # result
        pred = {'pdb': sample['pdb_info']['pdb'][0], }
        if self.distCB:
            pred['adj_pair'] = torch.stack((graph.all_edges()[0], graph.all_edges()[1]), -1).cpu().numpy()
            pred['distCB'] = F.softmax(output['distCB'], dim=-1).cpu().numpy()
        if self.QA:
            pred['global_lddt'] = float(output['global_lddt'].cpu().numpy())
            pred['local_lddt'] = output['local_lddt'].squeeze(1).cpu().numpy().tolist()

        return pred
        
    def pred(self, data_loader):
        print('Run Pred.')
        self.model.eval()
        with torch.no_grad():
            results = []
            for sample in data_loader:
                pdb_info = self.forward(sample)
                results.append(pdb_info)
                print(pdb_info['pdb'], 'done.')
            return results

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, default='/nfs_beijing_ai/jinxian/GNNRefine/data/pred_path/pred_path/casp15_af3.txt', help="path of starting model or folder containing starting models.")
    parser.add_argument("output", type=str, default='/nfs_beijing_ai/jinxian/GNNRefine/output/casp15_af3_Refine', help="path of output folder.")
    
    parser.add_argument('-n_decoy', type=int, dest='n_decoy', default=1, help='number of decoys built in each iteration.')
    parser.add_argument('-n_proc', type=int, dest='n_proc', default=1, help='number of processes running in parallel (>=1, recommendation is >=n_decoy).')
    parser.add_argument('-device_id', type=int, dest='device_id', default=0, help='device id (-1 for CPU, >=0 for GPU).')

    parser.add_argument("-save_qa", action="store_true", default=False, help="save QA results.")
    parser.add_argument("-save_le_decoy", action="store_true", default=False, help="save decoy models of lowest erergy in each iteration.")
    parser.add_argument("-save_all_decoy", action="store_true", default=False, help="save all decoy models.")
    parser.add_argument("-only_pred_dist", action="store_true", default=False, help="only predict the refined distance probability distribution.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)

    # set the CUDA_VISIBLE_DEVICES
    print("Using device:", args.device_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    if args.device_id>=0: args.device_id = 0
    
    assert args.n_proc>=1, 'n_proc must be >=1.'
    torch.set_num_threads(args.n_proc)

    if not args.only_pred_dist: from Folding import folding

    # prepare pdb
    # start_pdbs = []
    # print("input is:", args.input)
    # print("it is a file:", os.path.isfile(args.input))
    # print("it is a dir:", os.path.isdir(args.input))
    # if os.path.isfile(args.input):
    #     start_pdbs = [args.input, ]
    # elif os.path.isdir(args.input):
    #     for pdb in os.listdir(args.input):
    #         start_pdbs.append(args.input+'/'+pdb)
    # assert len(start_pdbs)>0, "start model not found."
    # if not os.path.isdir(args.output): os.mkdir(args.output)
    # print(start_pdbs, args.output)

    # 用新的path
    # start_pdbs = []
    file_path = args.input
    
    # 初始化一个空列表，用于存储文件中的数据
    start_pdbs = []

    # 使用with语句打开文件，确保文件会在操作完成后自动关闭
    with open(file_path, 'r', encoding='utf-8') as file:
        # 逐行读取文件
        for line in file:
            # 使用rstrip()移除行尾的换行符和其他空白字符
            clean_line = line.rstrip()
            # 将清理后的数据添加到列表中
            start_pdbs.append(clean_line)
    print('Todo ', len(start_pdbs), 'proteins')
    # Start_file_paths = os.listdir('/nfs_beijing_ai/jinxian/GNNRefine/data/af_v1m1')
    # for pro in Start_file_paths:
    #     if pro[-3:] == 'log':
    #         continue
    #     pro_file_path = os.listdir('/nfs_beijing_ai/jinxian/GNNRefine/data/af_v1m1/'+pro)
        
    #     for pdb_file in pro_file_path:
    #         if (pdb_file[-4:] == '.pdb' or  pdb_file[-4:] == '.cif'):
    #             start_pdbs.append('/nfs_beijing_ai/jinxian/GNNRefine/data/af_v1m1/'+pro+'/'+pdb_file)
    print(start_pdbs, args.output)

    print("refinement")
    gnn_param_dir = "/nfs_beijing_ai/jinxian/GNNRefine/data/gnn_params/"
    gnn_params = ['model.1.pkl', 'model.2.pkl', 'model.3.pkl', 'model.DAN1.pkl', 'model.DAN2.pkl',]
    tmp_dir = tempfile.TemporaryDirectory(dir = "/tmp/")
    _pdbs, lowenergy_pdbs = start_pdbs.copy(), {}
    gnn_pred = GNNPred(args.device_id, )
    for gnn_param in gnn_params:
        print(gnn_param, 'start')
        # work_dir
        work_dir = "%s/%s/"%(tmp_dir.name, gnn_param.split('.pkl')[0])
        if not os.path.isdir(work_dir): os.mkdir(work_dir)
        # dataset
        data = Data(_pdbs)
        print("_pdbs is :  ", _pdbs)
        data_loader = DataLoader(data, pin_memory=True, num_workers=args.n_proc)
        # pred
        gnn_pred.load_model(gnn_param_dir+gnn_param)
        refined_dist = gnn_pred.pred(data_loader)
        # refined dist
        if args.only_pred_dist:
            refined_dist_file = "%s/refined_dist.%s.pkl"%(args.output, gnn_param.split('.pkl')[0])
            pickle.dump(refined_dist, open(refined_dist_file, "wb"))
            print("refined dist saved at: %s"%(refined_dist_file))
            continue
        else:
            refined_dist_file = "%s/refined_dist.pkl"%(work_dir)
            pickle.dump(refined_dist, open(refined_dist_file, "wb"))

        # folding and select low energy pdb
        _lowenergy_pdbs = folding(refined_dist_file, work_dir, n_decoy=args.n_decoy, n_proc=args.n_proc)
        lowenergy_pdbs[gnn_param] = _lowenergy_pdbs
        _pdbs = [_lowenergy_pdbs[_] for _ in _lowenergy_pdbs]
        print(_pdbs)
        print(gnn_param, 'done')

    if args.only_pred_dist: exit()

    print("QA")
    qa_gnn_param = 'model.QA.pkl'
    gnn_QA = GNNPred(args.device_id, distCB=False, QA=True)
    gnn_QA.load_model(gnn_param_dir+qa_gnn_param)
    for start_pdb in start_pdbs:
        le_pdbs, _pdb = [], start_pdb
        for gnn_param in gnn_params:
            _pdb = lowenergy_pdbs[gnn_param][_pdb]
            le_pdbs.append(_pdb)
        # QA
        data = Data(le_pdbs)
        print("QA data is : ", le_pdbs)
        data_loader = DataLoader(data, pin_memory=True, num_workers=args.n_proc)        
        qa_results = gnn_QA.pred(data_loader)
        # select pdb by QA
        selected_item = sorted(qa_results, key=lambda k: -k['global_lddt'])[0]
        refined_pdb = "%s/%s.refined.pdb"%(args.output, start_pdb.split('/')[-1])
        os.system("cp %s %s"%(selected_item['pdb'], refined_pdb))
        print('refined pdb:', refined_pdb)
        # save QA results
        if args.save_qa:
            selected_item['pdb'] = refined_pdb
            pickle.dump(selected_item, open("%s.qa.pkl"%(refined_pdb), "wb"))
        # save low energy decoy models
        if args.save_le_decoy:
            decoys_dir = "%s/lowenergy_decoys/"%(args.output)
            if not os.path.isdir(decoys_dir): os.mkdir(decoys_dir)
            os.system("cp %s %s/"%(' '.join(le_pdbs), decoys_dir))
            print("low energy decoys saved at %s"%(decoys_dir))
    
    # save all decoy models
    if args.save_all_decoy:
        decoys_dir = "%s/all_decoys/"%(args.output)
        if not os.path.isdir(decoys_dir): os.mkdir(decoys_dir)
        os.system("cp %s/*/*.pdb %s/"%(tmp_dir.name, decoys_dir))
        print("all decoys saved at %s"%(decoys_dir))

    tmp_dir.cleanup()
    print('all done.')