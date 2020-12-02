import numpy as np
import pdb
from scipy.optimize import fsolve
from scipy.interpolate import splprep, splev


def define_3phase_plane(data):
    A_vec = data[:, :-1, 0] - data[:, :-1, 2]
    B_vec = data[:, :-1, 1] - data[:, :-1, 2]
    return np.cross(A_vec, B_vec)

def make_3p_splines(data_3phase):
    if len(data_3phase) > 1:
        tck1, u1 = splprep([data_3phase[:, 0, 1],
                            data_3phase[:, 1, 1],
                            data_3phase[:, 2, 1],
                            data_3phase[:, 3, 1]])
        return tck1
    else:
        return None

def define_z_vec(z, data):
    vec = data[:, :-1, 1] - z[:-1][np.newaxis, :]
    return vec

def find_gamma(gamma, z, phase_info):
    if type(gamma) == np.float:
        output = np.zeros(1)
        gamma = np.asarray([gamma])
    else:
        output = np.zeros(len(gamma))
    use_inds = (gamma<1.0)&(gamma>0.0)
    output[~use_inds] = 1e2
    if np.sum(use_inds) > 0:
        gamma_use = gamma[use_inds]
        tie_tri_data = phase_info['lookup'](gamma_use)
        tie_tri_vec = define_3phase_plane(tie_tri_data)
        z_vec = define_z_vec(z, tie_tri_data)
        output[use_inds] = np.sum(tie_tri_vec * z_vec, axis=1)
        output[use_inds] += 1e2*((gamma[use_inds] < 0) | (gamma[use_inds] > 1))
    return output

def gamma_cross(z, phase_info):
    gamma = fsolve(find_gamma, 0.5, args=(z, phase_info), factor=0.1, maxfev=20)
    return gamma

def tie_tri_func(z, phase_info):
    min_h2o = phase_info['data'][:, 0, :].min()
    max_h2o = phase_info['data'][:, 0, :].max()
    # pdb.set_trace()
    if (z[0] < min_h2o) or (z[0] > max_h2o):
        empty_result = np.zeros([4, 3])
        return empty_result, empty_result, 1e6, 1e6
    if (z == 0.0).any():
        # pdb.set_trace()
        get_ind = np.sum((z == 0.0) == (np.sum(phase_info['data'],axis=2) == 0.0), axis=1) == 4
        if get_ind.any() == 0.0:
            empty_result = np.zeros([4, 3])
            return empty_result, empty_result, 1e6, 1e6
        else:
            # pdb.set_trace()
            tie_tri_data = phase_info['data'][get_ind, :, :][0, :, :]
            if phase_info['inds'] == [0, 2, 4]:
                # pdb.set_trace()
                tie_tri_data
    else:
        gamma = gamma_cross(z, phase_info)
        tie_tri_data = np.squeeze(phase_info['lookup'](gamma))
        tie_tri_data = np.minimum(1.0, np.maximum(0.0, tie_tri_data))
        tie_tri_data = tie_tri_data / np.sum(tie_tri_data, axis=0)[np.newaxis, :]
    alpha = np.linalg.lstsq(tie_tri_data, z)[0]
    z_error = np.linalg.norm(np.dot(tie_tri_data, alpha) - z)
    alpha_error = np.linalg.norm(1.0 - np.sum(alpha))
    # if z_error < 1e-4:
    #     pdb.set_trace()
    return tie_tri_data, alpha, z_error, alpha_error

def tie_line_func(z, phase_info, ntake=5, p=3, dist_w_h2o=False, rach_rice_eps=1e-5):
    tie_lns = np.zeros([len(z), 2])
    z = z / np.sum(z)
    P0 = np.squeeze(phase_info['data'][:, :, 0])
    P1 = np.squeeze(phase_info['data'][:, :, 1])
    min_h2o = phase_info['data'][:, 0, :].min()
    max_h2o = phase_info['data'][:, 0, :].max()
    z_bounds = (z[0] < min_h2o) or (z[0] > max_h2o)
    zero_mask = (z == 0.0)
    keep_inds = (np.sum(P0[:, ~zero_mask], axis=1) == 1.0)
    z_missing = ~(zero_mask[np.newaxis, :] == (P1 == 0)).all(axis=1).any()
    if (np.sum(keep_inds) == 0.0) or (z_missing):
        empty_result = np.zeros([np.sum(zero_mask), np.sum(zero_mask)])
        return empty_result, empty_result, 1e6, 1e6
    elif z_bounds:
        # pdb.set_trace()
        small_ind = phase_info['data'][0, 0, :].argmin()
        big_ind = phase_info['data'][0, 0, :].argmax()
        alpha_z = np.zeros(2)
        tie_lns = np.zeros((len(z), 2))
        error = 1e-5
        if (z[0] < min_h2o) and (min_h2o < 0.01):
            tie_lns[:, small_ind] = z
            alpha_z[small_ind] = 1
        elif (z[0] > max_h2o) and (max_h2o > 0.9):
            tie_lns[:, big_ind] = z
            alpha_z[big_ind] = 1
        else:
            tie_lns[:, big_ind] = z
            alpha_z[big_ind] = 1
            error += 1e6
        return tie_lns, alpha_z, error, error
    else:
        P0, P1 = P0[keep_inds, :][:, ~zero_mask], P1[keep_inds, :][:, ~zero_mask]
        K = P0 / np.maximum(1e-20, P1)
        P0_z_dist = z[np.newaxis, ~zero_mask] - P0
        P_dist = P1 - P0
        alpha_guess = (np.sum(P0_z_dist * P_dist, axis=1)
                    / np.sum(P_dist ** 2, axis=1))
        if dist_w_h2o:
            dist = np.sqrt(np.sum(
                    (P0_z_dist
                     - alpha_guess[:, np.newaxis] * P_dist) **2, axis=1))
        else:
            dist = np.sqrt(np.sum(
                    (P0_z_dist[:, 1:]
                     - alpha_guess[:, np.newaxis] * P_dist[:, 1:]) **2, axis=1))
        ind_sort = np.argsort(dist)
        ind_take = ind_sort[:ntake]
        P0_ntake = P0[ind_take, :]
        P1_ntake = P1[ind_take, :]
        dist_ntake = np.maximum(1e-20, dist[ind_take])
        dist_weight = (1 / dist_ntake ** p) / np.sum(1 / dist_ntake ** p)
        z_P0 = np.sum(P0_ntake * dist_weight[:, np.newaxis], axis=0)
        z_P1 = np.sum(P1_ntake * dist_weight[:, np.newaxis], axis=0)
        z_out = np.stack([z_P0, z_P1]).T
        alpha_z = np.linalg.lstsq(z_out, z[~zero_mask])[0]
        alpha_z /= np.sum(alpha_z)
        z_error = np.linalg.norm(np.dot(z_out, alpha_z) - z[~zero_mask])
        alpha_inbounds = ~((alpha_z < 0).any() & (alpha_z > 1).any())
        # if ((z_error > 1e-6)
        #     and (alpha_inbounds)
        #     and ((alpha_z.min() > z.min()/10)
        #         or len(K) < 10)):
        # mismatch = np.max((np.abs(z_P1 - P1).min(0), np.abs(z_P0 - P0).min(0)))
        if ((z_error > 1e-6)
            and (alpha_inbounds)):
            # pdb.set_trace()
            proj_alpha = alpha_z[0]
            K_z = z_P0 / z_P1
            alpha_eps = 1e6
            Res = 1e6
            iter_alpha = 0
            while (np.abs(alpha_eps) > 1e-6) and (np.abs(Res) > 1e-6) and (iter_alpha < 5):
                Res = np.sum(z[~zero_mask]*(K_z - 1)/(1 + proj_alpha*(K_z - 1)))
                JRes_alpha = -np.sum((z[~zero_mask]*(K_z - 1)**2)/(1 + proj_alpha*(K_z - 1))**2)
                proj_alpha += - Res / JRes_alpha
                alpha_eps = np.abs(Res / JRes_alpha) / proj_alpha
                iter_alpha += 1
            z_P1 = z[~zero_mask] / (1 + (K_z - 1)*proj_alpha)
            z_P0 = K_z * z_P1
            # pdb.set_trace()
            # new_mismatch = np.max((np.abs(z_P1 - P1).min(0), np.abs(z_P0 - P0).min(0)))
            if (np.abs(alpha_z[0] - proj_alpha) < rach_rice_eps):
                alpha_z = np.asarray([proj_alpha, 1.0 - proj_alpha])
                alpha_z /= np.sum(alpha_z)
                z_out = np.stack([z_P0, z_P1]).T
                z_error = np.linalg.norm(np.dot(z_out, alpha_z) - z[~zero_mask])
            # pdb.set_trace()
        tie_lns[~zero_mask, :]= z_out
        alpha_error = np.linalg.norm(1.0 - np.sum(alpha_z))
        if alpha_z.min() < 0:
            z_error *= 1/2
            alpha_error *= 1/2
        return tie_lns, alpha_z, z_error, alpha_error

def tie_quad_func(z, phase_info):
    tie_quad = phase_info['data']
    alpha_z = np.linalg.lstsq(tie_quad, z)[0]
    z_error = np.linalg.norm(np.dot(tie_quad, alpha_z) - z)
    alpha_error = np.linalg.norm(1.0 - np.sum(alpha_z))
    return tie_quad, alpha_z, z_error, alpha_error

def create_lookup(tck1, data):
    def threephase_lookup(gamma):
        if tck1 is not None:
            phase1 = np.squeeze(np.transpose(np.asarray(splev(gamma, tck1))))
            index = np.interp(phase1[1], data[:, 1, 1], np.arange(len(data)))
            phase0 = np.transpose(np.asarray(
                [np.interp(index, np.arange(len(data)), data[:, ii, 0]) for ii in range(4)]))
            phase2 = np.transpose(np.asarray(
                [np.interp(index, np.arange(len(data)), data[:, ii, 2]) for ii in range(4)]))
            return np.dstack([phase0, phase1, phase2])
        else:
            return data
    return threephase_lookup


class quick_lookup(object):

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.phase_4_dict = data_dict['4_phase']
        self.phase_3_dict = data_dict['3_phase']
        self.phase_2_dict = data_dict['2_phase']
        self.h2o_stats = data_dict['h2o_stats']
        self.insert_3phase_lookup()

    def insert_3phase_lookup(self):
        for key in self.phase_3_dict.keys():
            data_3phase = self.phase_3_dict[key]['data']
            tck0 = make_3p_splines(data_3phase)
            mat_lookup = create_lookup(tck0, data_3phase)
            self.phase_3_dict[key]['spline'] = (tck0)
            self.phase_3_dict[key]['lookup'] = mat_lookup

    def flash(self, z, do_2phase_first=False):
        if (z == 0).all() or (np.sum(np.asarray(z)) == 0):
            pdb.set_trace()
        z /= np.sum(np.asarray(z))
        error = 1e6
        error_min = 1e6
        check_4phase = True
        check_2phase = False
        phase_inds = []
        phases_gtone = []
        key = 0
        output = ()
        z_comps = np.sum(z != 0.0)
        temp_result = ([], ([], [], [], []), None)
        if z[0] < self.h2o_stats['min']:
            output = ([2], (z, np.ones(1), 1e-6, 1e-6), 1e-6)

        elif z[0] > self.h2o_stats['max']:
            output = ([0], (z, np.ones(1), 1e-6, 1e-6), 1e-6)
        else:
            if z_comps < 4:
                check_4phase = False
                if (z_comps < 3) or (do_2phase_first):
                    check_2phase = True
            while error > 1e-6:
                phases_gtone = []
                if (check_4phase) and (key < len(self.phase_4_dict)):
                    phase_inds = self.phase_4_dict[key]['inds']
                    temp_result = tie_quad_func(z, self.phase_4_dict[key])
                elif (check_4phase) and (key == len(self.phase_4_dict)):
                    check_4phase = False
                    if do_2phase_first:
                        check_2phase = True
                    else:
                        check_2phase = False
                    key = -1
                elif (not check_2phase) and (not check_4phase) and (key < len(self.phase_3_dict)):
                    phase_inds = self.phase_3_dict[key]['inds']
                    temp_result = tie_tri_func(z, self.phase_3_dict[key])
                elif (not do_2phase_first) and (not check_2phase) and (not check_4phase) and (key == len(self.phase_3_dict)):
                    check_2phase = True
                    key = -1
                elif (do_2phase_first) and (check_2phase) and (not check_4phase) and (key == len(self.phase_2_dict)):
                    check_2phase = False
                    key = -1
                elif (check_2phase) and (not check_4phase) and (key < len(self.phase_2_dict)):
                    phase_inds = self.phase_2_dict[key]['inds']
                    temp_result = tie_line_func(z, self.phase_2_dict[key])
                else:
                    break
                key += 1
                error = np.sum(temp_result[-2:])
                neg_inds = temp_result[1] < 0
                if (neg_inds.any()):
                    phases_gtone = np.asarray(phase_inds)[~neg_inds]

                if (len(phase_inds) > 2) or \
                        (3 in phases_gtone) or \
                        (4 in phases_gtone):
                    error += 1 * (
                        (temp_result[0] > 1).any()
                        or (temp_result[0] < 0).any()
                        or (temp_result[1] > 1).any()
                        or (temp_result[1] < 0).any())
                if error < error_min:
                    output = (phase_inds, temp_result, error)
                    error_min = output[-1]

            # pdb.set_trace()
            if output == ():
                pdb.set_trace()
                print('There is an issue!')
            neg_inds = output[1][1] < 0
            if (neg_inds.any()):
                if (len(output[0]) == 2):
                    new_result = list()
                    if neg_inds[0]:
                        new_result.append(np.vstack((np.zeros(4), z)).T)
                    else:
                        new_result.append(np.vstack((z, np.zeros(4))).T)
                    new_result.append(np.minimum(1.0, np.maximum(0.0, output[1][1])))
                    new_result.append(temp_result[2])
                    new_result.append(3)
                    output = (output[0], new_result, output[-1])


        return output




