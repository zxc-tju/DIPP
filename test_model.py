import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
from utils.train_utils import MFMA_loss, motion_metrics
from model.planner import MotionPlanner
from model.predictor import Predictor
from torch.utils.data import DataLoader
from utils.test_utils import *
from waymo_open_dataset.protos import scenario_pb2
from torch.utils.data import Dataset
from torch.nn import functional as F


class Inter_DrivingData(Dataset):
    def __init__(self, file_list):

        self.data_list = file_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        retries = 3  # 定义重试次数
        for attempt in range(retries):
            try:
                data = np.load(self.data_list[idx])
                ego = data['ego']
                neighbors = data['neighbors']
                ref_line = data['ref_line']
                map_lanes = data['map_lanes']
                map_crosswalks = data['map_crosswalks']
                gt_future_states = data['gt_future_states']
                # ego_gt_control = data['ego_gt_control']
                # ego_control_acceleration = data['ego_control_acceleration']
                # ego_control_yaw_rate = data['ego_control_yaw_rate']
                # ego_gt_control = np.column_stack((ego_control_acceleration, ego_control_yaw_rate))
                # return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states, ego_gt_control
                return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states
            except Exception as e:
                print(f"Attempt {attempt+1} - Error loading {self.data_list[idx]}: {e}")
                if attempt == retries - 1:
                    return None
                
def test_model(data_loader, predictor, planner, use_planning, device):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(device)
        neighbors = batch[1].to(device)
        map_lanes = batch[2].to(device)
        map_crosswalks = batch[3].to(device)
        ref_line_info = batch[4].to(device)
        ground_truth = batch[5].to(device)
        future_action = ground_truth[:, 0, :, :2]
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)

        # predict
        with torch.no_grad():
            plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks,future_action)
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
            loss = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights)  # multi-future multi-agent loss

        # plan
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                "control_variables": plan.view(-1, 100),  # generate initial control sequence
                "predictions": prediction,  # generate predictions for surrounding vehicles
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i + 1}'] = cost_function_weights[:, i].unsqueeze(1)

            with torch.no_grad():
                final_values, info = planner.layer.forward(planner_inputs)

            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            # plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
            plan_cost = planner.objective.error_metric().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3])
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost  # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show progress
        current += batch[0].shape[0]
        sys.stdout.write(
            f"\rTesting Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time() - start_time) / current:>.4f}s/sample")
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    # predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    # epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    # logging.info(
    #     f'\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')
    predictorADE, predictorFDE_5, predictorFDE_3, predictorFDE_1 = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3]), np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE_5, predictorFDE_3, predictorFDE_1 ]
    print('\nModel Type: ', predictor.structure)
    print(
        f'test-plannerADE: {plannerADE:.4f}, test-plannerFDE: {plannerFDE:.4f} \ntest-predictorADE: {predictorADE:.4f} \ntest-predictorFDE_5: {predictorFDE_5:.4f}, test-predictorFDE_3: {predictorFDE_3:.4f}, test-predictorFDE_1: {predictorFDE_1:.4f}')

    return np.mean(epoch_loss), epoch_metrics


def read_file_to_list(file_path):
    with open(file_path,'r') as file:
        return [line.strip() for line in file.readlines()]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(50,future_model='DIPP').to(device)
    model_path = 'training_log/DIPP_10_percent_step_1_2024-07-12_11-28-05/model_40_0.8287.pth'
    predictor.load_state_dict(torch.load(model_path, map_location=device))
    trajectory_len, feature_len = 50, 9
    planner = MotionPlanner(trajectory_len, feature_len, device)

    test_files_list = read_file_to_list('training_log/training_data_log/DIPP_10_percent_step_1_2024-07-12_11-28-05/selected_files_val.txt')
    # # Remove this substring
    # remove_substring = '/home/zxc/Documents/data/Waymo_sample/processed_normalized_10percent'
    # # Add this substring in front
    # add_substring = '/home/liuyiru/Dataset/Waymo/processed_step_1_10percent'

    # test_files_list = [add_substring + filename.replace(remove_substring, '')
    #                     for filename in test_files_list]

    test_set = Inter_DrivingData(test_files_list)

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=1)
    # 评估模型
    mean_loss, metrics = test_model(test_loader, predictor, planner, use_planning=True, device=device)
    # print(f'Mean Loss: {mean_loss}')
    # print(f'Metrics: {metrics}')
