import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
from utils.train_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from torch.utils.data import DataLoader
from utils.test_utils import *
from waymo_open_dataset.protos import scenario_pb2


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
            f"\rValid Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time() - start_time) / current:>.4f}s/sample")
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    # predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    # epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    # logging.info(
    #     f'\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')
    predictorADE, predictorFDE_5, predictorFDE_3, predictorFDE_1 = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3]), np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE_5, predictorFDE_3, predictorFDE_1 ]
    logging.info(
        f'\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE_5: {predictorFDE_5:.4f}, val-predictorFDE_3: {predictorFDE_3:.4f}, val-predictorFDE_1: {predictorFDE_1:.4f}')

    return np.mean(epoch_loss), epoch_metrics
def open_loop_test(device):

    files = glob.glob('/home/liuyiru/git_code/DIPP_v1.0/DIPP/data/training_20s' + '/*')
    processor = TestDataProcess()

    # cache results
    collisions = []
    red_light, off_route = [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE_1, prediction_FDE_3, prediction_FDE_5 = [], [], [], []

    # load model
    predictor = Predictor(50).to(device)
    predictor.load_state_dict(torch.load('/home/liuyiru/git_code/DIPP_v0.0/DIPP/training_log/DIPP/model_30_1.6092.pth', map_location=device))
    predictor.eval()

    trajectory_len, feature_len = 50, 9
    planner = MotionPlanner(trajectory_len, feature_len, device=device, test=True)

    # iterate test files
    for file in files:
        scenarios = tf.data.TFRecordDataset(file)

        # iterate scenarios in the test file
        for scenario in scenarios:
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())

            scenario_id = parsed_data.scenario_id
            if scenario_id == 'cf610deef4e786e6':

                sdc_id = parsed_data.sdc_track_index
                timesteps = parsed_data.timestamps_seconds

                # build map
                processor.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

                # get a testing scenario
                for timestep in range(20, len(timesteps)-50, 10):
                    # prepare data
                    input_data = processor.process_frame(timestep, sdc_id, parsed_data.tracks)

                    ego = torch.from_numpy(input_data[0]).to(device)
                    neighbors = torch.from_numpy(input_data[1]).to(device)
                    lanes = torch.from_numpy(input_data[2]).to(device)
                    crosswalks = torch.from_numpy(input_data[3]).to(device)
                    ref_line = torch.from_numpy(input_data[4]).to(device)
                    neighbor_ids, norm_gt_data, gt_data = input_data[5], input_data[6], input_data[7]
                    current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

                    # predict
                    with torch.no_grad():
                        plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, lanes, crosswalks)
                        plan, prediction = select_future(plans, predictions, scores)

                    # plan
                    if True:
                        planner_inputs = {
                            "control_variables": plan.view(-1, 100),
                            "predictions": prediction,
                            "ref_line_info": ref_line,
                            "current_state": current_state
                        }

                        for i in range(feature_len):
                            planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(0)

                        with torch.no_grad():
                            final_values, info = planner.layer.forward(planner_inputs, optimizer_kwargs={'track_best_solution': True})
                            plan = info.best_solution['control_variables'].view(-1, 50, 2).to(device)

                    plan = bicycle_model(plan, ego[:, -1])[:, :, :3]
                    plan = plan.cpu().numpy()[0]

                    # compute metrics

                    # collision = check_collision(plan, norm_gt_data[1:], current_state.cpu().numpy()[0, :, 5:])
                    collision = False
                    collisions.append(collision)
                    traffic = check_traffic(plan, ref_line.cpu().numpy()[0])
                    red_light.append(traffic[0])
                    off_route.append(traffic[1])
                    # logging.info(f"Collision: {collision}, Red light: {traffic[0]}, Off route: {traffic[1]}")

                    # Acc, Jerk, Lat_Acc = check_dynamics(plan)
                    Acc, Jerk, Lat_Acc = 0,0,0
                    Accs.append(Acc)
                    Jerks.append(Jerk)
                    Lat_Accs.append(Lat_Acc)
                    # logging.info(f"Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

                    # Acc, Jerk, Lat_Acc = check_dynamics(norm_gt_data[0])
                    Human_Accs.append(Acc)
                    Human_Jerks.append(Jerk)
                    Human_Lat_Accs.append(Lat_Acc)
                    logging.info(f"Human: Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

                    # similarity = check_similarity(plan, norm_gt_data[0])
                    # similarity_1s.append(similarity[9])
                    # similarity_3s.append(similarity[29])
                    # similarity_5s.append(similarity[49])
                    # logging.info(f"Similarity@1s: {similarity[9]}, Similarity@3s: {similarity[29]}, Similarity@5s: {similarity[49]}")

                    prediction_error = check_prediction(prediction[0].cpu().numpy(), norm_gt_data[:,1:])
                    prediction_ADE.append(prediction_error[0])
                    prediction_FDE_1.append(prediction_error[1])
                    prediction_FDE_3.append(prediction_error[2])
                    prediction_FDE_5.append(prediction_error[3])
                    logging.info(f"Prediction ADE: {prediction_error[0]}, FDE: {prediction_error[1]}")


def read_file_to_list(file_path):
    with open(file_path,'r') as file:
        return [line.strip() for line in file.readlines()]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(50,future_model='DIPP').to(device)
    model_path = '/home/zxc/Downloads/model_40_DIPP.pth'
    predictor.load_state_dict(torch.load(model_path, map_location=device))
    trajectory_len, feature_len = 50, 9
    planner = MotionPlanner(trajectory_len, feature_len, device)

    test_files_list = read_file_to_list('training_log/training_data_log/CrossTransformer_v2_10_percent_step_1_2024-07-10_10-58-14/selected_files_test.txt')
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
    print(f'Mean Loss: {mean_loss}')
    print(f'Metrics: {metrics}')
