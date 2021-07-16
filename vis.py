import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

setting = 'informer_aiops_ftsr_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0' 

preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')

shape = preds.shape
print(shape)
vis_dir = './results/'+setting+'/vis'
os.makedirs(vis_dir, exist_ok=True)

for i in range(shape[0]//1000):
    # draw prediction
    idx = i * 1000 + np.random.randint(1000)
    fig = plt.figure()
    plt.plot(trues[idx,:,-1], label='GroundTruth')
    plt.plot(preds[idx,:,-1], label='Prediction')
    plt.legend()
    fig.savefig('{}/{}.png'.format(vis_dir, idx), dpi=fig.dpi)


