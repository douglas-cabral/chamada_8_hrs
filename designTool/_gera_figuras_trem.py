# -*- coding: utf-8 -*-
"""
Gera as figuras dos angulos criticos do trem de pouso (tipback, tailstrike e
overturn) a partir da geometria real da aeronave modelada no designTool.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

from designTool.standard_airplane import standard_airplane
from designTool.analyze import analyze
from designTool.plots import plot_geometry

# ----------------------------------------------------------------------------
# Roda a analise para pegar a geometria e os angulos exatos
air = standard_airplane('my_airplane')
analyze(air, print_log=False)

inp = air['inputs']
bal = air['balance']
lg = air['landing_gear']

x_nlg = inp['x_nlg']
x_mlg = inp['x_mlg']
y_mlg = inp['y_mlg']
z_lg = inp['z_lg']
x_ts = inp['x_tailstrike']
z_ts = inp['z_tailstrike']
D_f = inp['D_f']
L_f = inp['L_f']
xcg_fwd = bal['xcg_fwd']
xcg_aft = bal['xcg_aft']

a_tip = np.degrees(lg['alpha_tipback'])
a_ts = np.degrees(lg['alpha_tailstrike'])
a_ot = np.degrees(lg['phi_overturn'])

OUT = os.path.join(os.path.dirname(__file__), 'figuras_trem')
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Perfil lateral da fuselagem (mesma parametrizacao usada em plots.py)
xx = np.array([0.0, 1.24/41.72, 3.54/41.72, 7.55/41.72, x_ts/L_f, 1.0])
hh = np.array([0.0, 2.27/4.0, 3.56/4.0, 1.0, 1.0, 1.07/4.0])


def fuse_profile(n=400):
    xs, ztop, zbot = [], [], []
    for i in range(len(xx)-1):
        xseg = np.linspace(xx[i], xx[i+1], n)*L_f
        hseg = np.linspace(hh[i], hh[i+1], n)*D_f
        for xc, hc in zip(xseg, hseg):
            yye = (D_f-hc)/2 if xc > x_ts else 0.0
            xs.append(xc)
            ztop.append(yye + hc/2)
            zbot.append(yye - hc/2)
    return np.array(xs), np.array(ztop), np.array(zbot)


xf, ztop, zbot = fuse_profile()


def draw_wheel(ax, xc, zc, r=0.85):
    ax.add_patch(Circle((xc, zc), r, facecolor='0.55', edgecolor='k', lw=1.3, zorder=5))
    ax.add_patch(Circle((xc, zc), r*0.35, facecolor='0.85', edgecolor='k', lw=0.8, zorder=6))


# ============================================================================
# FIGURA 1 - TIPBACK
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.fill_between(xf, zbot, ztop, color='#cfe0f2', edgecolor='#1f4e79', lw=1.4, zorder=2)

# solo
ax.plot([x_nlg-3, x_mlg+8], [z_lg, z_lg], color='saddlebrown', lw=1.6, zorder=1)

# rodas
draw_wheel(ax, x_nlg, z_lg+0.85)
draw_wheel(ax, x_mlg, z_lg+0.85)
ax.annotate('NLG', (x_nlg, z_lg-0.1), ha='center', va='top', fontsize=9, fontweight='bold')
ax.annotate('MLG', (x_mlg, z_lg-0.1), ha='center', va='top', fontsize=9, fontweight='bold')

# CG traseiro
ax.plot([xcg_aft], [0.0], marker='o', ms=9, mfc='k', mec='k', zorder=7)
ax.annotate(r'CG traseiro', (xcg_aft, 0.0), textcoords='offset points',
            xytext=(-10, 12), ha='right', fontsize=9)

# vertice do angulo = contato do MLG com o solo
vx, vz = x_mlg, z_lg
# semirreta 1: vertical (referencia)
ax.plot([vx, vx], [vz, 3.2], 'k--', lw=1.1, zorder=4)
# semirreta 2: contato do MLG -> CG traseiro
ax.plot([vx, xcg_aft], [vz, 0.0], color='crimson', lw=1.8, zorder=4)

# arco do angulo (entre a vertical e a reta ate o CG), junto ao vertice
r_arc = 3.6
th_vert = 90.0
th_line = np.degrees(np.arctan2(0.0-vz, xcg_aft-vx))
ax.add_patch(Arc((vx, vz), 2*r_arc, 2*r_arc, angle=0,
                 theta1=th_vert, theta2=th_line, color='crimson', lw=2.0, zorder=6))
# valor do angulo (um pouco mais para cima para nao ser cortado pela fuselagem)
ax.annotate(r'$\alpha_{tb}=%.1f^\circ$' % a_tip,
            (vx+0.5, vz+3.7), color='crimson', fontsize=12, fontweight='bold', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

# cota nariz -> CG traseiro (linha com setas nas duas pontas + valor em cima)
z_dim = 4.6
ax.plot([0.0, 0.0], [0.15, z_dim+0.35], color='0.35', lw=0.8, zorder=3)
ax.plot([xcg_aft, xcg_aft], [0.15, z_dim+0.35], color='0.35', lw=0.8, zorder=3)
ax.annotate('', xy=(xcg_aft, z_dim), xytext=(0.0, z_dim),
            arrowprops=dict(arrowstyle='<->', color='0.20', lw=1.4))
ax.annotate(r'$x_{cg,aft}=%.2f$ m' % xcg_aft, (xcg_aft/2, z_dim+0.15),
            ha='center', va='bottom', fontsize=10)

# cota nariz -> MLG (na parte de baixo, abaixo da aeronave)
z_dimb = -7.0
ax.plot([0.0, 0.0], [0.0, z_dimb-0.35], color='0.35', lw=0.8, zorder=3)
ax.plot([x_mlg, x_mlg], [z_lg, z_dimb-0.35], color='0.35', lw=0.8, zorder=3)
ax.annotate('', xy=(x_mlg, z_dimb), xytext=(0.0, z_dimb),
            arrowprops=dict(arrowstyle='<->', color='0.20', lw=1.4))
ax.annotate(r'$x_{mlg}=%.2f$ m' % x_mlg, (x_mlg/2, z_dimb+0.15),
            ha='center', va='bottom', fontsize=10)

# cota da posicao vertical do trem (valor a esquerda da linha de setas)
xg = 2.4
ax.plot([0.0, xg], [0.0, 0.0], color='0.35', lw=0.7, ls=':', zorder=3)
ax.annotate('', xy=(xg, z_lg), xytext=(xg, 0.0),
            arrowprops=dict(arrowstyle='<->', color='0.20', lw=1.4))
ax.text(xg-0.5, z_lg/2, r'$-z_{lg}=5{,}70$ m', ha='center', va='center',
        rotation=90, fontsize=10)

ax.set_title(r'Ângulo de tipback: $\alpha_{tb}=%.1f^\circ$' % a_tip, fontsize=11)
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_aspect('equal')
ax.set_xlim(x_nlg-5, L_f*0.72)
ax.set_ylim(z_dimb-1.0, 6.2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'angulo_tipback.png'), dpi=250)
plt.close(fig)

# ============================================================================
# FIGURA 2 - TAILSTRIKE (vista lateral do run_airplane.py, com zoom)
# ============================================================================
tmp_png = os.path.join(OUT, '_tmp_zx.png')
figt, axt = plot_geometry(air, figname=tmp_png, az1=0, az2=-90, show=False)
axt.set_proj_type('ortho')
axt.view_init(0, -90)

# ponto de tailstrike identificado por legenda (texto na regiao vazia)
axt.scatter([x_ts], [0.0], [z_ts], s=90, color='crimson', edgecolors='k',
            linewidths=0.8, zorder=20, label='ponto de tailstrike')

# representacao do angulo (arco) na parte inclinada da cauda, com raio grande
R_ts = 11.0
tt = np.linspace(0.0, np.radians(a_ts), 40)
axt.plot(x_mlg + R_ts*np.cos(tt), np.zeros_like(tt), z_lg + R_ts*np.sin(tt),
         color='crimson', lw=2.0, zorder=20)
axt.text(x_mlg + (R_ts+2.4)*np.cos(np.radians(a_ts/2)), 0.0,
         z_lg + (R_ts+2.4)*np.sin(np.radians(a_ts/2)),
         r'$\alpha_{ts}=%.1f^\circ$' % a_ts, color='crimson',
         fontsize=13, fontweight='bold', ha='left', zorder=21)

axt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.16), fontsize=10,
           framealpha=0.95, borderpad=0.6)

# zoom na regiao da cauda mantendo escala igual em x e z
x0, x1 = 29.5, L_f+1.5
z0, z1 = z_lg-0.8, 2.2
axt.set_xlim3d(x0, x1)
axt.set_ylim3d(-2.0, 2.0)
axt.set_zlim3d(z0, z1)
axt.set_box_aspect(((x1-x0), 4.0, (z1-z0)))

# limpa o eixo y (profundidade) mantendo os ticks para nao quebrar o layout 3D
axt.set_yticklabels([])
axt.set_ylabel('')
axt.set_xlabel('x [m]', labelpad=14)
axt.set_zlabel('z [m]')

axt.set_title(r'Ângulo de tailstrike: $\alpha_{ts}=%.1f^\circ$' % a_ts, fontsize=11)
figt.savefig(os.path.join(OUT, 'angulo_tailstrike.png'), dpi=250, bbox_inches='tight')
plt.close(figt)
if os.path.exists(tmp_png):
    os.remove(tmp_png)

# ============================================================================
# FIGURA 3 - OVERTURN (2 paineis: vista superior + triangulo do angulo)
# ============================================================================
sgl = (xcg_fwd - x_nlg)*y_mlg/np.sqrt((x_mlg-x_nlg)**2 + y_mlg**2)
h_lg = -z_lg

fig, (axp, axr) = plt.subplots(1, 2, figsize=(12, 5),
                               gridspec_kw={'width_ratios': [1.6, 1]})

# ---- Painel (a): vista superior ----
axp.fill_between([0, L_f], [-D_f/2, -D_f/2], [D_f/2, D_f/2],
                 color='#eef3fa', edgecolor='0.6', lw=1.0, zorder=1)
for (xc, yc, lb) in [(x_nlg, 0, 'NLG'), (x_mlg, y_mlg, 'MLG'), (x_mlg, -y_mlg, 'MLG')]:
    axp.add_patch(Circle((xc, yc), 0.9, facecolor='0.55', edgecolor='k', lw=1.2, zorder=5))
    axp.annotate(lb, (xc, yc), textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=9, fontweight='bold')

axp.plot([x_nlg, x_mlg, x_mlg, x_nlg], [0, y_mlg, -y_mlg, 0],
         color='0.3', lw=1.2, zorder=3)
axp.plot([x_nlg, x_mlg], [0, y_mlg], color='navy', lw=2.0, zorder=4)

axp.plot([xcg_fwd], [0], marker='o', ms=9, mfc='k', mec='k', zorder=6)
axp.annotate('CG dianteiro', (xcg_fwd, 0), textcoords='offset points',
             xytext=(0, -16), ha='center', fontsize=9)

dx, dy = (x_mlg-x_nlg), (y_mlg-0)
t = ((xcg_fwd-x_nlg)*dx + (0-0)*dy)/(dx*dx+dy*dy)
px, py = x_nlg+t*dx, 0+t*dy
axp.plot([xcg_fwd, px], [0, py], color='crimson', lw=1.8, zorder=5)
axp.annotate(r'$s_{gl}=%.2f$ m' % sgl, ((xcg_fwd+px)/2, (0+py)/2),
             textcoords='offset points', xytext=(6, 6), color='crimson',
             fontsize=10, fontweight='bold')

axp.set_title('(a) Vista superior', fontsize=10)
axp.set_xlabel('x [m]')
axp.set_ylabel('y [m]')
axp.set_aspect('equal')
axp.set_xlim(-2, L_f*0.62)
axp.set_ylim(-y_mlg-3, y_mlg+3)
axp.grid(alpha=0.25)

# ---- Painel (b): triangulo do angulo de overturn ----
# vertices: contato do pneu (sgl,0); base do CG (0,0); CG (0,h_lg)
axr.plot([0, sgl], [0, 0], color='saddlebrown', lw=1.8)   # chao (reta horizontal)
axr.plot([0, 0], [0, h_lg], 'k--', lw=1.2)                # altura do trem
axr.plot([sgl, 0], [0, h_lg], color='crimson', lw=2.0)    # hipotenusa (MLG -> CG)
axr.plot([sgl], [0], marker='o', ms=8, mfc='0.55', mec='k', zorder=5)
axr.plot([0], [h_lg], marker='o', ms=8, mfc='k', mec='k', zorder=5)
axr.annotate('contato do pneu (MLG)', (sgl, 0), textcoords='offset points',
             xytext=(-4, 8), ha='right', fontsize=9)
axr.annotate('CG', (0, h_lg), textcoords='offset points',
             xytext=(8, 0), fontsize=9)
axr.annotate(r'$-z_{lg}=%.2f$ m' % h_lg, (0, h_lg/2),
             textcoords='offset points', xytext=(6, 0), fontsize=9)
axr.annotate(r'$s_{gl}=%.2f$ m' % sgl, (sgl/2, 0),
             textcoords='offset points', xytext=(0, -14), ha='center', fontsize=9)

# arco no VERTICE do contato do pneu, entre o chao e a hipotenusa
r_arc = 1.6
th2 = np.degrees(np.arctan2(h_lg, sgl))       # = phi_overturn
ax_th1 = 180.0 - th2
ax_th2 = 180.0
axr.add_patch(Arc((sgl, 0), 2*r_arc, 2*r_arc, angle=0,
                  theta1=ax_th1, theta2=ax_th2, color='crimson', lw=2.0, zorder=6))
thm = np.radians(180.0 - th2/2)
axr.annotate(r'$\phi_{ot}=%.1f^\circ$' % a_ot,
             (sgl + (r_arc+0.15)*np.cos(thm), (r_arc+0.15)*np.sin(thm)),
             color='crimson', fontsize=12, fontweight='bold', ha='right')

axr.set_title('(b) Plano do tombamento lateral', fontsize=10)
axr.set_xlabel(r'$s_{gl}$ [m]')
axr.set_ylabel(r'altura [m]')
axr.set_aspect('equal')
axr.set_xlim(-0.6, sgl+0.8)
axr.set_ylim(-0.8, h_lg+0.8)
axr.grid(alpha=0.25)

fig.suptitle(r'Ângulo de overturn: $\phi_{ot}=%.1f^\circ$' % a_ot, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, 'angulo_overturn.png'), dpi=250)
plt.close(fig)

print('sgl = %.3f m' % sgl)
print('Figuras salvas em:', OUT)
