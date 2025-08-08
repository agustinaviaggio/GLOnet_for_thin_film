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
        self.spectra = params.spectra
    
        self._init_simulation_parameters(params)
        self.n_bot = self.to_cuda_if_available(params.n_bot)  # number of frequencies or 1
        self.n_top = self.to_cuda_if_available(params.n_top)  # number of frequencies or 1
        self.k = self.to_cuda_if_available(params.k)  # number of frequencies
        self.theta = self.to_cuda_if_available(params.theta) # number of angles       
        self.pol = params.pol # str of pol
        self.target_spectra = self.to_cuda_if_available(params.target_spectra) if not self.sensor else None
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
                                            gamma=params.gamma)   

    def _init_simulation_parameters(self, params):
        if params.user_define:
            if self.sensor:
                self.n_database_empty = self.to_cuda_if_available(params.n_database_empty)
                self.n_database_full_A = self.to_cuda_if_available(params.n_database_full_A)
                self.n_database_full_B = self.to_cuda_if_available(params.n_database_full_B)
            else:
                self.n_database = self.to_cuda_if_available(params.n_database)
        else:
            if self.sensor:
                self.materials_empty = self.to_cuda_if_available(params.materials_empty)
                self.matdatabase_empty = self.to_cuda_if_available(params.matdatabase_empty)
                self.materials_full_A = self.to_cuda_if_available(params.materials_full_A)
                self.matdatabase_full_A = self.to_cuda_if_available(params.matdatabase_full_A)
                self.materials_full_B = self.to_cuda_if_available(params.materials_full_B)
                self.matdatabase_full_B = self.to_cuda_if_available(params.matdatabase_full_B)               
            else:
                self.matdatabase = self.to_cuda_if_available(params.matdatabase)
                self.materials = self.to_cuda_if_available(params.materials)

    def _create_spline(self, filename):
        df = pd.read_csv(filename, sep=';', decimal=',')
        df.columns = ['Wavelength [nm]', 'Spectral response']
        spline = UnivariateSpline(df['Wavelength [nm]'] / 1000, df['Spectral response'])
        spline.set_smoothing_factor(0.006)
        return spline

    def train(self):
        self.generator.train()
            
        # training loop
        with tqdm(total=self.numIter) as t:
            it = self.iter0
            final_loss = None 
            while True:
                it +=1 

                # normalized iteration number
                normIter = it / self.numIter

                # discretizaton coeff.
                self.update_alpha(normIter)
                
                # terminate the loop
                if it > self.numIter:
                    break 

                # sample z
                z = self.sample_z(self.batch_size)
                
                # generate a batch of images
                if self.sensor:
                    thicknesses, refractive_indices_empty, refractive_indices_full_A, refractive_indices_full_B, _ = self.generator(z, self.alpha)
                    #print(refractive_indices_full_A)
                    #print(refractive_indices_full_B)
                    #print(refractive_indices_full_B.shape)
                else:
                    thicknesses, refractive_indices, _ = self.generator(z, self.alpha)
                # calculate efficiencies and gradients using EM solver
                if self.sensor:
                    reflection_empty, transmission_empty = TMM_solver(thicknesses, refractive_indices_empty, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                    reflection_full_A, transmission_full_A = TMM_solver(thicknesses, refractive_indices_full_A, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                    reflection_full_B, transmission_full_B = TMM_solver(thicknesses, refractive_indices_full_B, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                else:
                    reflection, transmission = TMM_solver(thicknesses, refractive_indices, self.n_bot, self.n_top, self.k, self.theta, self.pol) 
                
                # free optimizer buffer 
                self.optimizer.zero_grad()

                # construct the loss
                if self.spectra:
                    sensor_signal = self.sensor_signal_2(self.k, reflection_empty, reflection_full_A, reflection_full_B) if self.sensor else None
                    g_loss = self.global_loss_function(sensor_signal) if self.sensor else self.global_loss_function(reflection)
                    FM = torch.pow(sensor_signal - 0.25, 2) if self.sensor else torch.pow(reflection - self.target_spectra, 2)
                else:
                    sensor_signal = self.sensor_signal_2(self.k, transmission_empty, transmission_full_A, transmission_full_B) if self.sensor else None
                    g_loss = self.global_loss_function(sensor_signal) if self.sensor else self.global_loss_function(transmission)
                    FM = torch.pow(sensor_signal - 0.25, 2) if self.sensor else torch.pow(transmission - self.target_spectra, 2)
                # record history
                self.record_history(it, g_loss, thicknesses, refractive_indices, FM) if not self.sensor else self.record_history(it, g_loss, thicknesses, refractive_indices_empty, FM)
                
                # train the generator
                g_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # update progress bar
                t.update()
                final_loss = g_loss.item() 
        return final_loss
    
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
            thicknesses, refractive_indices_empty, refractive_indices_full_A, refractive_indices_full_B, P = self.generator(z, self.alpha)
            result_mat = torch.argmax(P, dim=2).detach() # batch size x number of layer

            if not grayscale:
                ref_idx_empty, ref_idx_full_A, ref_idx_full_B = self._calculate_refractive_indices(kvector)
            else:
                if self.user_define:
                    ref_idx_empty, ref_idx_full_A, ref_idx_full_B = refractive_indices_empty, refractive_indices_full_A, refractive_indices_full_B
                else:
                    n_database_empty = self.to_cuda_if_available(self.matdatabase_empty.interp_wv(2 * math.pi/kvector, self.materials_empty, False).unsqueeze(0).unsqueeze(0))
                    ref_idx_empty = torch.sum(P.unsqueeze(-1) * n_database_empty, dim=2)
                    n_database_full_A = self.to_cuda_if_available(self.matdatabase_full_A.interp_wv(2 * math.pi/kvector, self.materials_full_A, False).unsqueeze(0).unsqueeze(0))
                    ref_idx_full_A = torch.sum(P.unsqueeze(-1) * n_database_full_A, dim=2)
                    n_database_full_B = self.to_cuda_if_available(self.matdatabase_full_B.interp_wv(2 * math.pi/kvector, self.materials_full_B, False).unsqueeze(0).unsqueeze(0))
                    ref_idx_full_B = torch.sum(P.unsqueeze(-1) * n_database_full_B, dim=2)
            
            reflection_empty, transmission_empty = TMM_solver(thicknesses, ref_idx_empty, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
            reflection_full_A, transmission_full_A = TMM_solver(thicknesses, ref_idx_full_A, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
            reflection_full_B, transmission_full_B = TMM_solver(thicknesses, ref_idx_full_B, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
            
            if self.spectra:
                sensor_signal = self.sensor_signal_2(self.to_cuda_if_available(kvector), reflection_empty, reflection_full_A, reflection_full_B)
            else:
                sensor_signal = self.sensor_signal_2(self.to_cuda_if_available(kvector), transmission_empty, transmission_full_A, transmission_full_B)
            return thicknesses, result_mat, sensor_signal, ref_idx_empty, reflection_empty, transmission_empty, ref_idx_full_A, reflection_full_A, reflection_full_A, ref_idx_full_B, reflection_full_B, reflection_full_B
        
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

            reflection, transmission = TMM_solver(thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
            return (thicknesses, ref_idx, result_mat, reflection, transmission)
      
    def _calculate_refractive_indices(self, result_mat, kvector):
        if self.user_define:
            n_database_empty = self.to_cuda_if_available(self.n_database_empty) # do not support dispersion
            n_database_full = self.to_cuda_if_available(self.n_database_full) # do not support dispersion
        else:
            n_database_empty = self.to_cuda_if_available(self.matdatabase_empty.interp_wv(2 * math.pi / kvector, self.materials_empty, False).unsqueeze(0).unsqueeze(0))
            n_database_full_A = self.to_cuda_if_available(self.matdatabase_full_A.interp_wv(2 * math.pi / kvector, self.materials_full_A, False).unsqueeze(0).unsqueeze(0))
            n_database_full_B = self.to_cuda_if_available(self.matdatabase_full_B.interp_wv(2 * math.pi / kvector, self.materials_full_B, False).unsqueeze(0).unsqueeze(0))
        one_hot = self.to_cuda_if_available(torch.eye(len(self.materials_empty)))
        one_hot_mat = one_hot[result_mat].unsqueeze(-1)
        ref_idx_empty = torch.sum(one_hot_mat * n_database_empty, dim=2)
        ref_idx_full_A = torch.sum(one_hot_mat * n_database_full_A, dim=2)
        ref_idx_full_B = torch.sum(one_hot_mat * n_database_full_B, dim=2)
        return ref_idx_empty, ref_idx_full_A, ref_idx_full_B
    
    def _TMM_solver(self, thicknesses, result_mat, kvector = None, inc_angles = None, pol = None):
        if self.sensor:
            if kvector is None:
                kvector = self.k
            if inc_angles is None:
                inc_angles = self.theta
            if pol is None:
                pol = self.pol  
            n_database_empty = self.matdatabase_empty.interp_wv(2 * math.pi/kvector, self.materials_empty, False).unsqueeze(0).unsqueeze(0)
            n_database_full_A = self.matdatabase_full.interp_wv(2 * math.pi/kvector, self.materials_full_A, False).unsqueeze(0).unsqueeze(0)
            n_database_full_B = self.matdatabase_full.interp_wv(2 * math.pi/kvector, self.materials_full_B, False).unsqueeze(0).unsqueeze(0)
            one_hot = torch.eye(len(self.materials_empty))
            one_hot_mat = one_hot[result_mat].unsqueeze(-1)
            ref_idx_empty = torch.sum(one_hot_mat * n_database_empty, dim=2)
            ref_idx_full_A = torch.sum(one_hot_mat * n_database_full_A, dim=2)
            ref_idx_full_B = torch.sum(one_hot_mat * n_database_full_B, dim=2)
            reflection_e, transmission_e = TMM_solver(thicknesses, ref_idx_empty, self.n_bot, self.n_top, kvector, inc_angles, pol)
            reflection_f_A , transmission_f_A= TMM_solver(thicknesses, ref_idx_full_A, self.n_bot, self.n_top, kvector, inc_angles, pol)
            reflection_f_B, transmission_f_B = TMM_solver(thicknesses, ref_idx_full_B, self.n_bot, self.n_top, kvector, inc_angles, pol)
            return reflection_e, reflection_f_A, reflection_f_B, transmission_e, transmission_f_A, transmission_f_B
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
            reflection, transmission = TMM_solver(thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles, pol)
            return reflection, transmission            
        
    def update_alpha(self, normIter):
        self.alpha = round(normIter/0.05) * self.alpha_sup + 1.
        
    def sample_z(self, batch_size):
        return self.to_cuda_if_available(torch.randn(batch_size, self.noise_dim, requires_grad=True))

    def spectra_int(self, spectra, k, dim):
        lambdas = 2*math.pi/self.k
        return torch.trapz(spectra, lambdas, dim= dim)
    
    def sensor_signal_1(self, k, spectra_empty, spectra_full):
        lambdas = 2 * math.pi / self.k
        led_x_ldr = self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas) * self.ldr_spline(lambdas)))
        
        signal_empty = torch.matmul(spectra_empty.squeeze(),torch.diag(led_x_ldr))
        signal_full = torch.matmul(spectra_full.squeeze(),torch.diag(led_x_ldr))
        signal_diff = signal_empty - signal_full
        int_led = self.spectra_int(self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas))), self.k, dim = 0)
        int_diff = self.spectra_int(signal_diff, self.k, dim = 1)
        sensor_signal= torch.abs(int_diff)/int_led
        return sensor_signal   

    def sensor_signal_2(self, k, spectra_empty, spectra_full_A, spectra_full_B):
        lambdas = 2 * math.pi / self.k
        led_x_ldr = self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas) * self.ldr_spline(lambdas)))
        int_led = self.spectra_int(self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas))), self.k, dim = 0)
        signal_empty = torch.matmul(spectra_empty.squeeze(),torch.diag(led_x_ldr))
        signal_empty_int = self.spectra_int(signal_empty, self.k, dim = 1)
        signal_A = torch.matmul(spectra_full_A.squeeze(),torch.diag(led_x_ldr))
        signal_A_int = self.spectra_int(signal_A, self.k, dim = 1)
        if torch.all(spectra_full_B == 1):
            print("Warning: spectra_full_B is all ones, using signal_empty for sensor_signal_2")
            signal_diff = signal_empty_int - signal_A_int  # Igual a sensor_signal_1 en este caso
        else:
            signal_B = torch.matmul(spectra_full_B.squeeze(),torch.diag(led_x_ldr))
            signal_B_int = self.spectra_int(signal_B, self.k, dim = 1)
            signal_diff = (signal_empty_int - signal_A_int) * (signal_A_int - signal_B_int) * (signal_B_int - signal_empty_int) / (int_led **3)
        #int_led = self.spectra_int(self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas))), self.k, dim = 0)
        #int_diff = self.spectra_int(signal_diff, self.k, dim = 1)
        #sensor_signal= torch.abs(signal_diff)/int_led
        sensor_signal= torch.abs(signal_diff)
        return sensor_signal 

    def global_loss_function(self, signal):
        return -torch.mean(torch.exp(-torch.mean(torch.pow(signal - self.target_spectra, 2), dim=(1,2,3))/self.sigma)) if not self.sensor else -torch.mean(torch.exp(-torch.pow(signal - 0.25, 2)/self.sigma))
   
    def global_loss_function_robust(self, spectra, thicknesses):
        metric = torch.mean(torch.pow(spectra - self.target_spectra, 2), dim=(1,2,3))
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
        
