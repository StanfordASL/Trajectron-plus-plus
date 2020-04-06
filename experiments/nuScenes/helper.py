import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as pe
from scipy.ndimage import rotate
import seaborn as sns

from model.model_registrar import ModelRegistrar
from model import Trajectron
from utils import prediction_output_to_trajectories

from scipy.integrate import cumtrapz

line_colors = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0']

cars = [plt.imread('icons/Car TOP_VIEW 375397.png'),
        plt.imread('icons/Car TOP_VIEW F05F78.png'),
        plt.imread('icons/Car TOP_VIEW 80CBE5.png'),
        plt.imread('icons/Car TOP_VIEW ABCB51.png'),
        plt.imread('icons/Car TOP_VIEW C8B0B0.png')]

robot = plt.imread('icons/Car TOP_VIEW ROBOT.png')


def load_model(model_dir, env, ts=3999):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    hyperparams['map_enc_dropout'] = 0.0
    if 'incl_robot_node' not in hyperparams:
        hyperparams['incl_robot_node'] = False

    stg = Trajectron(model_registrar, hyperparams,  None, 'cpu')

    stg.set_environment(env)

    stg.set_annealing_params()

    return stg, hyperparams


def plot_vehicle_nice(ax, predictions, dt, max_hl=10, ph=6, map=None, x_min=0, y_min=0):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    assert (len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.fdata, origin='lower', alpha=0.5)

    cmap = ['k', 'b', 'y', 'g', 'r']
    line_alpha = 0.7
    line_width = 0.2
    edge_width = 2
    circle_edge_width = 0.5
    node_circle_size = 0.3
    a = []
    i = 0
    node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
    for node in node_list:
        history = histories_dict[node] + np.array([x_min, y_min])
        future = futures_dict[node] + np.array([x_min, y_min])
        predictions = prediction_dict[node] + np.array([x_min, y_min])
        if node.type.name == 'VEHICLE':
            # ax.plot(history[:, 0], history[:, 1], 'ko-', linewidth=1)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--o',
                    linewidth=4,
                    markersize=3,
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

            for t in range(predictions.shape[2]):
                sns.kdeplot(predictions[0, :, t, 0], predictions[0, :, t, 1],
                            ax=ax, shade=True, shade_lowest=False,
                            color=line_colors[i % len(line_colors)], zorder=600, alpha=0.8)

            vel = node.get(np.array([ts_key]), {'velocity': ['x', 'y']})
            h = np.arctan2(vel[0, 1], vel[0, 0])
            r_img = rotate(cars[i % len(cars)], node.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi,
                           reshape=True)
            oi = OffsetImage(r_img, zoom=0.025, zorder=700)
            veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
            veh_box.zorder = 700
            ax.add_artist(veh_box)
            i += 1
        else:
            # ax.plot(history[:, 0], history[:, 1], 'k--')

            for t in range(predictions.shape[2]):
                sns.kdeplot(predictions[0, :, t, 0], predictions[0, :, t, 1],
                            ax=ax, shade=True, shade_lowest=False,
                            color='b', zorder=600, alpha=0.8)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])
            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)


def plot_vehicle_mm(ax, predictions, dt, max_hl=10, ph=6, map=None, x_min=0, y_min=0):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    assert (len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.fdata, origin='lower', alpha=0.5)

    cmap = ['k', 'b', 'y', 'g', 'r']
    line_alpha = 0.7
    line_width = 0.2
    edge_width = 2
    circle_edge_width = 0.5
    node_circle_size = 0.5
    a = []
    i = 0
    node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
    for node in node_list:
        history = histories_dict[node] + np.array([x_min, y_min])
        future = futures_dict[node] + np.array([x_min, y_min])

        predictions = prediction_dict[node] + np.array([x_min, y_min])
        if node.type.name == 'VEHICLE':
            for sample_num in range(prediction_dict[node].shape[1]):
                ax.plot(predictions[:, sample_num, :, 0], predictions[:, sample_num, :, 1], 'ko-',
                        zorder=620,
                        markersize=5,
                        linewidth=3, alpha=0.7)
        else:
            for sample_num in range(prediction_dict[node].shape[1]):
                ax.plot(predictions[:, sample_num, :, 0], predictions[:, sample_num, :, 1], 'ko-',
                        zorder=620,
                        markersize=2,
                        linewidth=1, alpha=0.7)


def plot_vehicle_nice_mv(ax, predictions, dt, max_hl=10, ph=6, map=None, x_min=0, y_min=0):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    assert (len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.fdata, origin='lower', alpha=0.5)

    cmap = ['k', 'b', 'y', 'g', 'r']
    line_alpha = 0.7
    line_width = 0.2
    edge_width = 2
    circle_edge_width = 0.5
    node_circle_size = 0.3
    a = []
    i = 0
    node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
    for node in node_list:
        h = node.get(np.array([ts_key]), {'heading': ['°']})[0, 0]
        history_org = histories_dict[node] + np.array([x_min, y_min])
        history = histories_dict[node] + np.array([x_min, y_min]) + 5 * np.array([np.cos(h), np.sin(h)])
        future = futures_dict[node] + np.array([x_min, y_min]) + 5 * np.array([np.cos(h), np.sin(h)])
        predictions = prediction_dict[node] + np.array([x_min, y_min]) + 5 * np.array([np.cos(h), np.sin(h)])
        if node.type.name == 'VEHICLE':
            for t in range(predictions.shape[2]):
                sns.kdeplot(predictions[0, :, t, 0], predictions[0, :, t, 1],
                            ax=ax, shade=True, shade_lowest=False,
                            color=line_colors[i % len(line_colors)], zorder=600, alpha=1.0)

            r_img = rotate(cars[i % len(cars)], node.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi,
                           reshape=True)
            oi = OffsetImage(r_img, zoom=0.08, zorder=700)
            veh_box = AnnotationBbox(oi, (history_org[-1, 0], history_org[-1, 1]), frameon=False)
            veh_box.zorder = 700
            ax.add_artist(veh_box)
            i += 1
        else:

            for t in range(predictions.shape[2]):
                sns.kdeplot(predictions[:, t, 0], predictions[:, t, 1],
                            ax=ax, shade=True, shade_lowest=False,
                            color='b', zorder=600, alpha=0.8)

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)


def plot_vehicle_nice_mv_robot(ax, predictions, dt, max_hl=10, ph=6, map=None, x_min=0, y_min=0):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    assert (len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.fdata, origin='lower', alpha=0.5)

    cmap = ['k', 'b', 'y', 'g', 'r']
    line_alpha = 0.7
    line_width = 0.2
    edge_width = 2
    circle_edge_width = 0.5
    node_circle_size = 0.3

    node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
    for node in node_list:
        h = node.get(np.array([ts_key]), {'heading': ['°']})[0, 0]
        history_org = histories_dict[node] + np.array([x_min, y_min]) + 5 / 2 * np.array(
            [np.cos(h), np.sin(h)])
        future = futures_dict[node] + np.array([x_min, y_min]) + 5 * np.array([np.cos(h), np.sin(h)])

        ax.plot(future[:, 0],
                future[:, 1],
                '--o',
                c='#F05F78',
                linewidth=4,
                markersize=3,
                zorder=650,
                path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

        r_img = rotate(robot, node.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi, reshape=True)
        oi = OffsetImage(r_img, zoom=0.08, zorder=700)
        veh_box = AnnotationBbox(oi, (history_org[-1, 0], history_org[-1, 1]), frameon=False)
        veh_box.zorder = 700
        ax.add_artist(veh_box)


def integrate(f, dx, F0=0.):
    N = f.shape[0]
    return F0 + np.hstack((np.zeros((N, 1)), cumtrapz(f, axis=1, dx=dx)))