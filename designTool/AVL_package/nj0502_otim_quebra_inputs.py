# NJ-0502 - novo otimo (nacele livre + water spray + SM verificada no AVL + quebra Yehudi)
# Substitui as entradas correspondentes de 'my_airplane' em designTool/standard_airplane.py.
# A quebra NAO e modelada pelo designTool: a geometria dela esta em nj0502_otim_quebra.avl.
# Gerado por estudo_quebra_asa/scripts/03_gera_avl_final.py.
import numpy as np

novos_inputs = {
    'S_w'    : 386.6198896111,
    'AR_w'   : 10.7187815231,
    'sweep_w' : 35.1062142893*np.pi/180,
    'xr_w'   : 19.5377576200,
    'Cht'    : 0.7000000000,
    'Lc_h'   : 4.7487735486,
    'Cvt'    : 0.0589341096,
    'Lb_v'   : 0.4951547768,
    'x_mlg'  : 31.7773652626,
    'y_mlg'  : 6.9500000000,
    'z_lg'   : -5.7523247120,
    'x_n'    : 19.5994199317,
    'y_n'    : 9.9200000000,
    'z_n'    : -2.9852790683,
}

# Quebra (BF interno vertical): c_raiz = 15.2060 m, x_BF = 34.7438 m, y_quebra = 14.6401 m
