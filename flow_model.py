import numpy as np
import pdb
import dill as pickle
import os
from scipy.optimize import fsolve, root, fmin_l_bfgs_b, fmin_slsqp


def density_mat(S, rho):
    return S * rho[np.newaxis, :]

def molar_conc(x, rho, S):
    return np.sum(x * density_mat(S, rho)[:, np.newaxis, :], axis=-1)

def molar_flux(x, rho, S, S_c=None, Cor_exp=None, mu=None, ):
    Flux = frac_flow(S, S_c=S_c, Cor_exp=Cor_exp, mu=mu)
    return np.sum(x * (Flux * rho[np.newaxis, :])[:, np.newaxis, :], axis=-1)

def frac_flow(S, S_c=None, Cor_exp=None, mu=None):
    Np = S.shape[-1]

    if len(S.shape) == 1:
        Sh_frac = np.sum(S[-2:])
        Nx = 1
    else:
        Sh_frac = np.sum(S[:, -2:], axis=-1)
        Nx = S.shape[0]

    if S_c is None:
        S_c = 0.1 * np.ones_like(S)
    else:
        S_c = np.asanyarray(S_c)
        S_c = S_c[np.newaxis, :] * np.ones_like(S)


    if Cor_exp is None:
        Cor_exp = 2*np.ones(Np)
    else:
        Cor_exp = np.asarray(Cor_exp)

    if mu is None:
        mu = np.ones(Np)

    f_g = corey_model(S, S_c, Sh_frac, Cor_exp, mu)
    return f_g

def corey_model(S, S_c, Sh_frac, Cor_exp, mu):
    S_eff = S / (1 - Sh_frac)[:, np.newaxis]
    if len(S.shape) == 1:
        S_eff[-2:] = 0
    else:
        S_eff[:, -2:] = 0
    k_ri = np.zeros_like(S)
    f_i = np.zeros_like(S)

    # Using simplest possible form
    for ii, nn in enumerate(Cor_exp):
        k_ri[:, ii] =  (np.maximum(S_eff[:, ii] - S_c[:, ii], 0) / (1 - np.sum(S_c, axis=-1)))**nn
        f_i[:, ii] = k_ri[:, ii] / mu[ii]
    f_i /= np.sum(f_i, axis=-1)[:, np.newaxis]
    return f_i

def sat_calc(alpha, rho, eps=1e-8):
    S_guess = alpha
    error = 1e6

    while error > eps:
        S = alpha * np.sum(S_guess * rho[np.newaxis, :], axis=-1)[:, np.newaxis] / rho[np.newaxis, :]
        S /=  np.sum(S, axis=-1)[:, np.newaxis]
        error = np.linalg.norm(S - S_guess)
        S_guess = S[:, :]
    return S

def flux(S, rho):
    return rho[np.newaxis, :] * frac_flow(S)

class model(object):

    def __init__(self, lookup_table, Nc=4, Np=5, Nx=100, x_bounds=(0, 1), id='temp'):
        self.Nc = Nc
        self.Nx = Nx
        self.Np = Np
        self.xmin = x_bounds[0]
        self.xmax = x_bounds[1]
        self.id = id
        self.lookup_tbl = lookup_table
        self.set_flag = False
        self.make_grid()

    def set_bndry(self, zL, zR, vL=1):
        self.zL = np.asarray(zL)
        self.zR = np.asarray(zR)
        self.vL = vL

    def set_density(self, rho):
        self.rho = np.asarray(rho)

    def set_viscosity(self, mu):
        self.mu = np.asarray(mu)

    def set_fracflow(self, S_c=None, Cor_exp=None):
        if S_c is None:
            S_c = 0.1 * np.ones(self.Np)

        if Cor_exp is None:
            Cor_exp = 2 * np.ones(self.Np)

        self.S_c = S_c
        self.Cor_exp = Cor_exp

    def make_grid(self):
        self.xc = np.linspace(self.xmin, self.xmax, self.Nx)
        self.dx = self.xc[1] - self.xc[0]

    def save_model(self, last_step):
        save_str = str(self.tstep) + 'step_' +  str(self.id) + '.pkl'
        new_dict = {key: value for key, value in self.__dict__.items()}
        new_dict['new_data'] = last_step
        pickle.dump(self.__dict__, open(save_str, 'wb'))
        self.results = last_step


    def set_model(self, zL, zR, rho, vL=1, mu=None, S_c=None, Cor_exp=None):
        if mu is None:
            mu = np.ones_like(rho)
        self.set_bndry(zL, zR, vL=vL)
        self.set_density(rho)
        self.set_viscosity(mu)
        self.set_fracflow(S_c, Cor_exp)
        self.set_flag = True

    def gen_new_fields(self, x, alpha):
        S = sat_calc(alpha, self.rho)
        G = molar_conc(x, self.rho, S)
        H = molar_flux(x, self.rho, S, S_c=self.S_c, Cor_exp=self.Cor_exp, mu=self.mu)
        return {'S': S, 'G': G, 'H': H}


    def get_current_calc(self, z, x, S, G, H):
        return {'z': z, 'x': x, 'S': S, 'G': G, 'H': H}

    def calc_front(self, z, eps=1e-4):
        z_chng = (np.abs(z - self.z_init) > eps).any(axis=-1)
        if z_chng.any():
            front = max(self.front, np.max(np.arange(self.Nx)[z_chng]) + 2)
            return front
        else:
            return self.front

    def avg_density(self, rho, S):
        return np.sum(S * rho[np.newaxis, :], axis=-1)

    def calc_vd(self, S, G, H, vd, rho, last_ind=None):
        if last_ind is None:
            last_ind = len(G[:, 0])
        vd = np.maximum(0,
                    (np.sum(G[1:last_ind, :], axis=-1)
                     - self.avg_density(rho, S[1:last_ind, :])
                     + self.cfl * vd[:last_ind-1] * np.sum(H[:last_ind-1, :], axis=-1))
                     / (self.cfl * np.sum(H[1:last_ind, :], axis=-1) + 1e-30))
        return vd

    def calc_flow_change(self, G, H, vd, last_ind=None):
        if last_ind is None:
            last_ind = len(G[:, 0])
        G_new = (G[1:last_ind, :]
                 - self.cfl
                 * (H[1:last_ind, :] * vd[1:last_ind, np.newaxis]
                    - H[:last_ind-1, :] * vd[:last_ind-1, np.newaxis]))
        return G_new

    def density_closure(self, vd, G, H):
        vd_guess = np.hstack((1.0, vd))
        G_guess = np.vstack((G[0, :], self.calc_flow_change(G, H, vd_guess)))
        G_sum = np.sum(np.maximum(0, G_guess), axis=-1)
        z_guess = G_guess / (G_sum[:, np.newaxis] + 1e-30)
        # z_guess[z_guess < 1e-20] = 0
        if ((z_guess == 0).all(-1)).any() or ((np.isnan(z_guess)).all(-1)).any():
            pdb.set_trace()
        x_mat_guess, alpha_mat_guess = self.flash(z_guess)
        S_calc = sat_calc(alpha_mat_guess, self.rho)
        S_guess = np.maximum(0, S_calc)
        density_error = (1 - G_sum / self.avg_density(self.rho, S_guess))**2
        # Residue = density_error[1:] + 1e3 * np.sum(np.abs(np.minimum(0, G_guess[1:, :])), axis=-1)
        vd_neg = vd < 0
        S_neg = (S_calc[1:, :] < 0).any(axis=1)
        Residue = density_error[1:]
        Residue[vd_neg] += 1e1 * vd[vd_neg]**2
        Residue[S_neg] += 1e1 * (S_calc[1:, :].min(axis=1))[S_neg]**2
        return Residue

    def jac_density_closure(self, vd, G, H, eps=1e-6):
        Residue = self.density_closure(vd, G, H)
        Jac = np.zeros((len(vd), len(vd)))
        for ii in range(len(vd)):
            eps_mat = np.zeros(len(vd))
            eps_mat[ii] = eps
            Jac[:, ii] = (self.density_closure(vd + eps_mat, G, H)
                          - self.density_closure(vd - eps_mat, G, H)) / (2*eps)
        return Jac

    def jac_res(self, vd, G, H, eps=1e-6):
        Residue = self.density_closure(vd, G, H)
        Jac = np.zeros((len(vd), len(vd)))
        for ii in range(len(vd)):
            eps_mat = np.zeros(len(vd))
            eps_mat[ii] = eps
            Jac[:, ii] = (self.density_closure(vd + eps_mat, G, H)
                          - self.density_closure(vd - eps_mat, G, H)) / (2*eps)
            # Jac[:, ii] = (self.density_closure(vd + eps_mat, G, H)
            #               - Residue) / (eps)
        return Residue, Jac

    def newton_solve(self, vd, G, H, eps=1e-10):
        error, J = self.jac_res(vd, G, H)
        if np.isnan(vd).any():
            pdb.set_trace()
        count = 0
        # pdb.set_trace()
        while (error > eps).any() and (count < 5):
            # pdb.set_trace()
            # J = self.jac_density_closure(vd, G, H)
            # print(vd)
            # print(error)
            # print(J)
            del_vd = -np.linalg.solve(J, error)
            if (np.isnan(del_vd).any()) or (np.isnan(vd).any()):
                pdb.set_trace()
            vd += del_vd
            vd = np.maximum(0, vd)
            # error = self.density_closure(vd, G, H)
            error, J = self.jac_res(vd, G, H)
            count += 1
        flag = count >= 20
        return vd, flag



    def flash(self, z_mat):
        z_mat = np.asarray(z_mat)
        if len(z_mat.shape) > 1:
            x_mat = np.zeros((len(z_mat), self.Nc, self.Np))
            alpha_mat = np.zeros((len(z_mat), self.Np))
            output = [self.lookup_tbl(z) for z in z_mat]
            for ii, result in enumerate(output):
                try:
                    x_mat[ii, :, result[0]] = result[1][0].T
                    alpha_mat[ii, result[0]] = result[1][1]
                except:
                    pdb.set_trace()
        else:
            x_mat = np.zeros((self.Nc, self.Np))
            alpha_mat = np.zeros(self.Np)
            result = self.lookup_tbl(z_mat)
            x_mat[:, result[0]] = result[1][0]
            alpha_mat[result[0]] = result[1][1]
        alpha_mat = np.minimum(1, np.maximum(0, alpha_mat))
        return x_mat, alpha_mat

    def run_model(self, t_end=1.0, cfl=0.01, save_int=10, vd_max=5):
        self.cfl  = cfl
        if self.set_flag:
            self.dt = self.dx * self.cfl
            self.tsteps = int(t_end / self.dt)
            step_error = 1e6
            self.front = 3


            vd = np.ones(self.Nx)
            vd[0] = self.vL

            z_bndry = np.vstack([self.zL, self.zR])
            x_bndry, alpha_bndry = self.flash(z_bndry)
            S_bndry = sat_calc(alpha_bndry, self.rho)
            G_bndry = molar_conc(x_bndry, self.rho, S_bndry)
            H_bndry = molar_flux(x_bndry, self.rho, S_bndry,
                                 S_c=self.S_c, Cor_exp=self.Cor_exp, mu=self.mu)
            z_init = np.ones((self.Nx, self.Nc)) * z_bndry[-1, :][np.newaxis, :]
            z_init[0] = z_bndry[0, :]
            self.z_init = z_init.copy()

            x_init = np.ones((self.Nx, self.Nc, self.Np)) * x_bndry[-1, :, :][np.newaxis, :, :]
            x_init[0, :] = x_bndry[0, :, :]

            S_init = np.ones((self.Nx, self.Np)) * S_bndry[-1, :][np.newaxis, :]
            S_init[0] = S_bndry[0, :]

            G_init = np.ones((self.Nx, self.Nc)) * G_bndry[-1, :][np.newaxis, :]
            G_init[0] = G_bndry[0, :]

            H_init = np.ones((self.Nx, self.Nc)) * H_bndry[-1, :][np.newaxis, :]
            H_init[0] = H_bndry[0, :]
            last_step = self.get_current_calc(z_init, x_init, S_init, G_init, H_init)

            for tstep in range(self.tsteps):
                self.front = self.calc_front(last_step['z'])
                self.tstep = tstep

                z = last_step['z']
                x = last_step['x']
                S = last_step['S']
                G = last_step['G']
                H = last_step['H']



                vd_guess = np.minimum(vd_max,
                                      np.maximum(0,
                                      self.calc_vd(S[:self.front, :],
                                        G[:self.front, :],
                                        H[:self.front, :],
                                        vd[:self.front],
                                        self.rho)))

                # pdb.set_trace()
                # self.flash([0.89584794, 0.10023233, 0.00391972, 0.])

                if np.isnan(vd_guess).any():
                    pdb.set_trace()

                # G_test = np.asarray([[0, 0, 0.4, 0.6], [0.8521, 0.9417, 0, 0], [0.8521, 0.9417, 0, 0]])
                # H_test = np.asarray([[0, 0, 0.4, 0.6], [0.0002, 0.9998, 0, 0], [0.0002, 0.9998, 0, 0]])
                # S_test = np.asarray([[0, 1.0, 0], [0, 0.8004, 0.1996], [0, 0.8004, 0.1996]])
                # rho_test = np.asarray([5.3188, 1.0, 4.9774])
                # vd_test = self.calc_vd(S_test,
                #                         G_test,
                #                         H_test,
                #                         vd[:self.front],
                #                         rho_test)
                # pdb.set_trace()
                # self.density_closure(vd_test, G_test[:self.front,:], H_test[:self.front,:])
                # output = root(self.density_closure,
                #               vd_guess,
                #               jac=self.jac_density_closure,
                #               args=(G[:self.front, :],
                #                     H[:self.front, :]),
                #               options={'factor':1.0})
                #
                # if not output['success']:
                #     output = root(self.density_closure,
                #                   np.ones(len(vd_guess)),
                #                   args=(G[:self.front, :],
                #                         H[:self.front, :]),
                #                   options={'factor': 0.1})
                # # output = fmin_slsqp(self.density_closure,
                # #               vd_guess,
                # #               args=(G[:self.front, :],
                # #                     H[:self.front, :]))
                # vd[1:self.front] = np.minimum(2.0, np.maximum(0, output['x']))
                output = self.newton_solve(vd_guess,
                                           G[:self.front, :],
                                           H[:self.front, :])
                vd[1:self.front] = np.minimum(vd_max,np.maximum(0,output[0]))

                if np.isnan(vd).any():
                    pdb.set_trace()
                    #
                # try:
                #     output = self.newton_solve(vd_guess,
                #                                G[:self.front, :],
                #                                H[:self.front, :])
                    # if not output[1]:
                    #     vd[1:self.front] = np.minimum(vd_max,np.maximum(0,output[0]))
                    # else:
                    #     output = root(self.density_closure,
                    #                   np.ones(len(vd_guess)),
                    #                   args=(G[:self.front, :],
                    #                         H[:self.front, :]),
                    #                   options={'factor': 1.0})
                    #     vd[1:self.front] = np.minimum(vd_max,
                    #                                   np.maximum(0,
                    #                                              output['x']))
                # except:
                    # output = root(self.density_closure,
                    #               np.ones(len(vd_guess)),
                    #               args=(G[:self.front, :],
                    #                     H[:self.front, :]),
                    #               options={'factor': 1.0})
                    # vd[1:self.front] = np.minimum(vd_max,
                    #                               np.maximum(0,
                    #                                          output['x']))


                # if ((vd > 2.0).any()) or ((vd < 0).any()):
                    # pdb.set_trace()
                    # print('In this statement now')

                G[1:self.front, :] = np.maximum(
                    0,
                    self.calc_flow_change(G[:self.front],
                                          H[:self.front],
                                          vd[:self.front]))
                G_sum = np.sum(G[1:self.front, :], axis=-1)
                z_new = G[1:self.front] / (G_sum[:, np.newaxis] + 1e-30)
                z_new[z_new < 1e-20] = 0
                x_mat_new, alpha_mat_new = self.flash(z_new)
                S_new = np.maximum(0, sat_calc(alpha_mat_new, self.rho))
                H_new = np.maximum(0, molar_flux(x_mat_new,
                                                 self.rho,
                                                 S_new,
                                                 S_c=self.S_c,
                                                 Cor_exp=self.Cor_exp,
                                                 mu=self.mu))
                z[1:self.front, :] = z_new
                x[1:self.front, :, :] = x_mat_new
                S[1:self.front, :] = S_new
                H[1:self.front, :] = H_new

                if (tstep == 0) or (tstep % save_int == 0):
                    self.save_model(last_step)
                    print(S[:self.front, :])
                    print(z[:self.front, :])
                    print(vd[:self.front])

                last_step = self.get_current_calc(z, x, S, G, H)
                # print(tstep)
                # pdb.set_trace()


                # if (self.front == 20) or (tstep > 1000):
                #     # pdb.set_trace()
                #     print('In this statement')

                if (z_new.flatten() < 0).any():
                    pdb.set_trace()



