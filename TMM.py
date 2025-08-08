import torch

def transfer_matrix_layer(thickness, refractive_index, k, ky, pol):
    '''
    args:
        thickness (tensor): batch size x 1 x 1 x 1
        refractive_index (tensor): batch size x (number of frequencies or 1) x 1 x 1
        k (tensor): 1 x number of frequencies x 1 x 1
        ky (tensor): 1 x number of frequencies x number of angles x 1
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    kx = torch.sqrt(torch.pow(k * refractive_index, 2)  - torch.pow(ky, 2))

    TEpol = -torch.pow(refractive_index, 2)  #chequear
    TMpol = torch.ones_like(TEpol)

    if pol == 'TM':
        pol_multiplier = TMpol
    elif pol == 'TE':
        pol_multiplier = TEpol
    else:
        pol_multiplier = torch.cat([TMpol, TEpol], dim = -1)
    
    pol_dim = torch.ones_like(pol_multiplier)

    T11 = torch.cos(kx * thickness)*pol_dim

    T12 = - torch.sin(kx * thickness) * k / kx * pol_multiplier * 1j
    
    T21 = - torch.sin(kx * thickness) * kx / k * pol_multiplier * 1j
    
    T22 = torch.cos(kx * thickness)*pol_dim
    
    return T11, T12, T21, T22

def transfer_matrix_stack(thicknesses, refractive_indices, k, ky, pol = 'TM'):
    '''
    args:
        thickness (tensor): batch size x number of layers
        refractive_indices (tensor): batch size x number of layers x (number of frequencies or 1)
        k (tensor): 1 x number of frequencies x 1
        ky (tensor): 1 x number of frequencies x number of angles 
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    N = thicknesses.size(-1)
    numfreq = refractive_indices.size(-1)
    batch_size = thicknesses.size(0)
    num_angles = ky.size(2)  

    if pol in ['TM', 'TE']:
        num_pol = 1
    elif pol == 'both':
        num_pol = 2

    T_stack = torch.eye(2, 2, dtype=torch.complex64).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    T_stack = T_stack.repeat(batch_size, numfreq, num_angles, num_pol, 1, 1)

    for i in range(N):
        thickness = thicknesses[:, i].view(-1, 1, 1, 1)
        refractive_index = refractive_indices[:, i, :].view(-1, numfreq, 1, 1)

        T11, T12, T21, T22 = transfer_matrix_layer(thickness, refractive_index, k, ky, pol)
        T_layer = torch.stack((T11, T12, T21, T22), dim=-1).view(batch_size, numfreq, num_angles, num_pol, 2, 2)

        T_stack = torch.matmul(T_stack, T_layer)
        
    return T_stack

def amp2field(refractive_index, k, ky, pol = 'TM'):
    '''
    args:
        refractive_index (tensor): 1 x (number of frequencies or 1) x 1 x 1
        k (tensor): 1 x number of frequencies x 1 x 1
        ky (tensor): 1 x number of frequencies x number of angles x 1
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): 1 x number of frequencies x number of angles x number of pol
    '''
    kx = torch.sqrt(torch.pow(k * refractive_index, 2)  - torch.pow(ky, 2))

    TEpol = -torch.pow(refractive_index, 2)
    TMpol = torch.ones_like(TEpol)

    if pol == 'TM':
        pol_multiplier = TMpol
    elif pol == 'TE':
        pol_multiplier = TEpol
    elif pol == 'both':
        pol_multiplier = torch.cat([TMpol, TEpol], dim = -1)

    pol_dim = torch.ones_like(pol_multiplier)
    
    T11 = torch.ones_like(kx)*pol_dim
    T11 = T11.to(torch.complex64)

    T12 = torch.ones_like(kx)*pol_dim
    T12 = T12.to(torch.complex64)

    T21 = kx / k / pol_multiplier
    T21 = T21.to(torch.complex64)

    T22 = -kx / k / pol_multiplier
    T22 = T22.to(torch.complex64)

    numfreq = kx.size(1)
    num_angles = ky.size(2)
    if pol in ['TM', 'TE']:
        num_pol = 1
    elif pol == 'both':
        num_pol = 2

    # Stack them into a 2x2 complex matrix
    
    T = torch.stack((T11, T12, T21, T22), dim=-1).view(1, numfreq, num_angles, num_pol, 2, 2)

    return T

def TMM_solver(thicknesses, refractive_indices, n_bot, n_top, k, theta, pol = 'TM'):
    '''
    args:
        thickness (tensor): batch size x number of layers
        refractive_indices (tensor): batch size x number of layers x (number of frequencies or 1)
        k (tensor): number of frequencies
        theta (tensor): number of angles
        n_bot (tensor): 1 or number of frequencies
        n_top (tensor): 1 or number of frequencies
        pol (str): 'TM' or 'TE' or 'both'
     
    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    # adjust the format
    n_bot = n_bot.view(1, -1, 1, 1)
    n_top = n_top.view(1, -1, 1, 1)
    k = k.view(1, -1, 1, 1)
    ky = k * n_bot * torch.sin(theta.view(1, 1, -1, 1))

    if not refractive_indices.numel() == 0:
        # transfer matrix calculation
        T_stack = transfer_matrix_stack(thicknesses, refractive_indices, k, ky, pol)
        
        # amplitude to field convertion
        A2F_bot = amp2field(n_bot, k, ky, pol)
        A2F_top = amp2field(n_top, k, ky, pol)
        
        # S matrix
        S_stack = torch.matmul(torch.inverse(A2F_top), torch.matmul(T_stack, A2F_bot))
        
        # reflection 
        Reflection = torch.pow(torch.abs(S_stack[:,:,:,:,0,1]), 2) / torch.pow(torch.abs(S_stack[:,:,:,:,0,0]), 2)
        Reflection = Reflection.double()
        Transmission = torch.pow(torch.det(torch.abs(S_stack)), 2) / torch.pow(torch.abs(S_stack[:,:,:,:,0,0]), 2)
        Transmission = Transmission.double()
    else:
        if pol in ['TM', 'TE']:
            num_pol = 1
        elif pol == 'both':
            num_pol = 2
        Reflection = torch.ones(thicknesses.size(0), k.size(1), len(theta), num_pol)
        Reflection = Reflection.double()
        Transmission = torch.ones(thicknesses.size(0), k.size(1), len(theta), num_pol)
        Transmission = Transmission.double()
            
    return Reflection, Transmission