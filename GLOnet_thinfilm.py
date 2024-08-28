import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import UnivariateSpline
from TMM import *
from tqdm import tqdm
from net import Generator, ResGenerator

class GLOnet():
    def __init__(self, params):
        # GPU 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self._init_generator(params)   
        self.optimizer = self._init_optimizer(params)
        self.scheduler = self._init_scheduler(params)
                
        # training parameters
        self.noise_dim = params.noise_dim
        self.numIter = params.numIter
        self.batch_size = params.batch_size
        self.sigma = params.sigma
        self.alpha_sup = params.alpha_sup
        self.iter0 = 0
        self.alpha = 0.1
    
        # simulation parameters
        self.user_define = params.user_define
        if params.sensor:
            self.sensor = True
        else:
            self.sensor = False       
        self._init_simulation_parameters(params)
        self.n_bot = self.to_cuda_if_available(params.n_bot)  # number of frequencies or 1
        self.n_top = self.to_cuda_if_available(params.n_top)  # number of frequencies or 1
        self.k = self.to_cuda_if_available(params.k)  # number of frequencies
        self.theta = self.to_cuda_if_available(params.theta) # number of angles       
        self.pol = params.pol # str of pol
        self.target_reflection = self.to_cuda_if_available(params.target_reflection) if not self.sensor else None
        # 1 x number of frequencies x number of angles x (number of pol or 1)

        if self.sensor:
            self.led_spline = self._create_spline("true-green-osram.csv")
            self.ldr_spline = self._create_spline("ldr.csv")
        
        self.ruta = params.ruta
        self.seed = params.seed
        # tranining history
        self.loss_training = []
        self.refractive_indices_training = []
        self.thicknesses_training = []
        self.FM_training = []
        
    def to_cuda_if_available(self, tensor):
        if torch.cuda.is_available():
            return tensor.cuda()
        return tensor   

    def _init_generator(self, params):
        if params.net == 'Res':
            generator = ResGenerator(params)
        elif params.net == 'Dnn':
            generator = Generator(params)
        return generator.to(self.device)
    
    def _init_optimizer(self, params):
        return torch.optim.Adam(self.generator.parameters(), lr=params.lr, 
                                betas=(params.beta1, params.beta2), 
                                weight_decay=params.weight_decay)
    
    def _init_scheduler(self, params):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                            step_size=params.step_size, 
                                            gamma=params.step_size)   

    def _init_simulation_parameters(self, params):
        if params.user_define:
            if self.sensor:
                self.n_database_empty = self.to_cuda_if_available(params.n_database_empty)
                self.n_database_full = self.to_cuda_if_available(params.n_database_full)
            else:
                self.n_database = self.to_cuda_if_available(params.n_database)
        else:
            if self.sensor:
                self.materials_empty = self.to_cuda_if_available(params.materials_empty)
                self.matdatabase_empty = self.to_cuda_if_available(params.matdatabase_empty)
                self.materials_full = self.to_cuda_if_available(params.materials_full)
                self.matdatabase_full = self.to_cuda_if_available(params.matdatabase_full)
            else:
                self.matdatabase = self.to_cuda_if_available(params.matdatabase)
                self.materials = self.to_cuda_if_available(params.materials)

    def _create_spline(self, filename):
        df = pd.read_csv(filename, sep=';', decimal=',')
        df.columns = ['Wavelength [nm]', 'Reflection spectra']
        spline = UnivariateSpline(df['Wavelength [nm]'] / 1000, df['Reflection spectra'])
        spline.set_smoothing_factor(0.006)
        return spline

    def train(self):
        self.generator.train()
            
        # training loop
        with tqdm(total=self.numIter) as t:
            it = self.iter0  
            while True:
                it +=1 

                # normalized iteration number
                normIter = it / self.numIter

                # discretizaton coeff.
                self.update_alpha(normIter)
                
                # terminate the loop
                if it > self.numIter:
                    return 

                # sample z
                z = self.sample_z(self.batch_size)
                
                # generate a batch of images
                if self.sensor:
                    thicknesses, refractive_indices_empty, refractive_indices_full, _ = self.generator(z, self.alpha)
                else:
                    thicknesses, refractive_indices, _ = self.generator(z, self.alpha)
                # calculate efficiencies and gradients using EM solver
                if self.sensor:
                    reflection_empty = TMM_solver(thicknesses, refractive_indices_empty, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                    reflection_full = TMM_solver(thicknesses, refractive_indices_full, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                else:
                    reflection = TMM_solver(thicknesses, refractive_indices, self.n_bot, self.n_top, self.k, self.theta, self.pol) 
                
                # free optimizer buffer 
                self.optimizer.zero_grad()

                # construct the loss 
                sensor_signal = self.sensor_signal(self.k, reflection_empty, reflection_full) if self.sensor else None
                
                g_loss = self.global_loss_function(sensor_signal) if self.sensor else self.global_loss_function(reflection)

                FM = torch.pow(sensor_signal - 1, 2) if self.sensor else torch.pow(reflection - self.target_reflection, 2)

                # record history
                self.record_history(it, g_loss, thicknesses, refractive_indices, FM) if not self.sensor else self.record_history(it, g_loss, thicknesses, refractive_indices_empty, FM)
                
                # train the generator
                g_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # update progress bar
                t.update()
    
    def evaluate(self, num_devices, kvector = None, inc_angles = None, pol = None, grayscale=True):
        if kvector is None:
            kvector = self.k
        if inc_angles is None:
            inc_angles = self.theta
        if pol is None:
            pol = self.pol            

        self.generator.eval()
        z = self.sample_z(num_devices)
        if self.sensor:
            thicknesses, refractive_indices_empty, refractive_indices_full, P = self.generator(z, self.alpha)
            result_mat = torch.argmax(P, dim=2).detach() # batch size x number of layer

            if not grayscale:
                ref_idx_empty, ref_idx_full = self._calculate_refractive_indices(kvector)
            else:
                if self.user_define:
                    ref_idx_empty, ref_idx_full = refractive_indices_empty, refractive_indices_full
                else:
                    n_database_empty = self.to_cuda_if_available(self.matdatabase_empty.interp_wv(2 * math.pi/kvector, self.materials_empty, False).unsqueeze(0).unsqueeze(0))
                    ref_idx_empty = torch.sum(P.unsqueeze(-1) * n_database_empty, dim=2)
                    n_database_full = self.to_cuda_if_available(self.matdatabase_full.interp_wv(2 * math.pi/kvector, self.materials_full, False).unsqueeze(0).unsqueeze(0))
                    ref_idx_full = torch.sum(P.unsqueeze(-1) * n_database_full, dim=2)
            
            reflection_empty = TMM_solver(thicknesses, ref_idx_empty, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
            reflection_full = TMM_solver(thicknesses, ref_idx_full, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
            
            sensor_signal = self.sensor_signal(self.to_cuda_if_available(kvector), reflection_empty, reflection_full)
            
            return thicknesses, result_mat, sensor_signal, ref_idx_empty, reflection_empty, ref_idx_full, reflection_full
        
        else:
            thicknesses, refractive_indices, P = self.generator(z, self.alpha)
            result_mat = torch.argmax(P, dim=2).detach() # batch size x number of layer
            if not grayscale:
                if self.user_define:
                    n_database = self.n_database # do not support dispersion
                else:
                    n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, False).unsqueeze(0).unsqueeze(0).type(self.dtype)
            
                one_hot = torch.eye(len(self.materials)).type(self.dtype)
                ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database, dim=2)
            else:
                if self.user_define:
                    ref_idx = refractive_indices
                else:
                    n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, False).unsqueeze(0).unsqueeze(0).type(self.dtype)
                    ref_idx = torch.sum(P.unsqueeze(-1) * n_database, dim=2)

            reflection = TMM_solver(thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
            return (thicknesses, ref_idx, result_mat, reflection)
      
    def _calculate_refractive_indices(self, result_mat, kvector):
        if self.user_define:
            n_database_empty = self.to_cuda_if_available(self.n_database_empty) # do not support dispersion
            n_database_full = self.to_cuda_if_available(self.n_database_full) # do not support dispersion
        else:
            n_database_empty = self.to_cuda_if_available(self.matdatabase_empty.interp_wv(2 * math.pi / kvector, self.materials_empty, False).unsqueeze(0).unsqueeze(0))
            n_database_full = self.to_cuda_if_available(self.matdatabase_full.interp_wv(2 * math.pi / kvector, self.materials_full, False).unsqueeze(0).unsqueeze(0))
        
        one_hot = self.to_cuda_if_available(torch.eye(len(self.materials_empty)))
        one_hot_mat = one_hot[result_mat].unsqueeze(-1)
        ref_idx_empty = torch.sum(one_hot_mat * n_database_empty, dim=2)
        ref_idx_full = torch.sum(one_hot_mat * n_database_full, dim=2)
        return ref_idx_empty, ref_idx_full
    
    def _TMM_solver(self, thicknesses, result_mat, kvector = None, inc_angles = None, pol = None):
        if self.sensor:
            if kvector is None:
                kvector = self.k
            if inc_angles is None:
                inc_angles = self.theta
            if pol is None:
                pol = self.pol  
            n_database_empty = self.matdatabase_empty.interp_wv(2 * math.pi/kvector, self.materials_empty, False).unsqueeze(0).unsqueeze(0)
            n_database_full = self.matdatabase_full.interp_wv(2 * math.pi/kvector, self.materials_full, False).unsqueeze(0).unsqueeze(0)
            one_hot = torch.eye(len(self.materials_empty))
            one_hot_mat = one_hot[result_mat].unsqueeze(-1)
            ref_idx_empty = torch.sum(one_hot_mat * n_database_empty, dim=2)
            ref_idx_full = torch.sum(one_hot_mat * n_database_full, dim=2)
            reflection_e = TMM_solver(thicknesses, ref_idx_empty, self.n_bot, self.n_top, kvector, inc_angles, pol)
            reflection_f = TMM_solver(thicknesses, ref_idx_full, self.n_bot, self.n_top, kvector, inc_angles, pol)
            return reflection_e, reflection_f
        else:
            if kvector is None:
                kvector = self.k
            if inc_angles is None:
                inc_angles = self.theta
            if pol is None:
                pol = self.pol  
            n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, False).unsqueeze(0).unsqueeze(0)
            one_hot = torch.eye(len(self.materials)).type(self.dtype)
            ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database, dim=2)
            reflection = TMM_solver(thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles, pol)
            return reflection            
        
    def update_alpha(self, normIter):
        self.alpha = round(normIter/0.05) * self.alpha_sup + 1.
        
    def sample_z(self, batch_size):
        return self.to_cuda_if_available(torch.randn(batch_size, self.noise_dim, requires_grad=True))

    def spectra_int(self, spectra, k, dim):
        lambdas = 2*math.pi/self.k
        return torch.trapz(spectra, lambdas, dim= dim)
    
    def sensor_signal(self, k, reflection_empty, reflection_full):
        lambdas = 2 * math.pi / self.k
        led_x_ldr = self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas) * self.ldr_spline(lambdas)))
        
        signal_empty = torch.matmul(reflection_empty.squeeze(),torch.diag(led_x_ldr))
        signal_full = torch.matmul(reflection_full.squeeze(),torch.diag(led_x_ldr))
        signal_diff = signal_empty - signal_full
        int_led = self.spectra_int(self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas))), self.k, dim = 0)
        int_diff = self.spectra_int(signal_diff, self.k, dim = 1)
        sensor_signal= torch.abs(int_diff)/int_led
        return sensor_signal   

    def global_loss_function(self, signal):
        return -torch.mean(torch.exp(-torch.mean(torch.pow(signal - self.target_reflection, 2), dim=(1,2,3))/self.sigma)) if not self.sensor else -torch.mean(torch.exp(-torch.pow(signal - 1, 2)/self.sigma))
        
    def global_loss_function_robust(self, reflection, thicknesses):
        metric = torch.mean(torch.pow(reflection - self.target_reflection, 2), dim=(1,2,3))
        dmdt = torch.autograd.grad(metric.mean(), thicknesses, create_graph=True)
        return -torch.mean(torch.exp((-metric - self.robust_coeff *torch.mean(torch.abs(dmdt[0]), dim=1))/self.sigma))

    def record_history(self, it, loss, thicknesses, refractive_indices, FM):
        self.loss_training.append(loss.detach().numpy())
        if it == self.numIter:
            self.thicknesses_training.append(thicknesses.detach().numpy())
            self.refractive_indices_training.append(refractive_indices.detach().numpy())
            self.FM_training.append(FM.detach().numpy())
        
    def viz_training(self):
        plt.figure(figsize = (20, 5))
        plt.subplot(131)
        plt.plot(self.loss_training)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(str(self.ruta)+'/seed_'+str(self.seed)+'/loss.png', dpi=300)
        np.savez(str(self.ruta)+'/seed_'+str(self.seed)+'/loss', self.loss_training)
        np.savez(str(self.ruta)+'/seed_'+str(self.seed)+'/thicknesses', self.thicknesses_training)
        np.savez(str(self.ruta)+'/seed_'+str(self.seed)+'/ref_idxs', self.refractive_indices_training)
        np.savez(str(self.ruta)+'/seed_'+str(self.seed)+'/FM', self.FM_training)
        
