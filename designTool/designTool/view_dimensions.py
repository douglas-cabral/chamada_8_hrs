# -*- coding: utf-8 -*-
"""
Cotas dimensionais nas vistas ortograficas do aviao (padrao do tipback:
linhas de chamada + setas <-> + valor).
"""
import numpy as np
from mpl_toolkits.mplot3d import proj3d


def _proj(ax, xyz):
    x, y, z = xyz
    xp, yp, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return xp, yp


def _line3d(ax, p1, p2, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], **kwargs)


def _dim_segment(ax, p1, p2, e1, e2, label, label_va='bottom', label_ha='center',
                 label_offset=(0.0, 0.12)):
    """Cota entre p1 e p2 com linhas de chamada e1->p1, e2->p2."""
    _line3d(ax, e1, p1, color='0.35', lw=0.8, zorder=10)
    _line3d(ax, e2, p2, color='0.35', lw=0.8, zorder=10)

    x1, y1 = _proj(ax, p1)
    x2, y2 = _proj(ax, p2)
    ax.annotate(
        '',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle='<->', color='0.20', lw=1.3),
        xycoords='data',
        zorder=11,
    )

    mx, my = _proj(ax, (
        0.5 * (p1[0] + p2[0]),
        0.5 * (p1[1] + p2[1]),
        0.5 * (p1[2] + p2[2]),
    ))
    ax.annotate(
        label,
        (mx + label_offset[0], my + label_offset[1]),
        ha=label_ha,
        va=label_va,
        fontsize=9,
        color='0.15',
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                  edgecolor='none', alpha=0.85),
        zorder=12,
    )


def add_planview_dimensions(ax, airplane, view):
    """
    Adiciona cotas principais a uma vista ortografica.

    view: 'top' (XY), 'side' (ZX) ou 'front' (YZ)
    """
    inp = airplane['inputs']
    geo = airplane['geometry']

    L_f = inp['L_f']
    D_f = inp['D_f']
    b_w = geo['b_w']
    xr_w = inp['xr_w']
    x_mlg = inp['x_mlg']
    y_mlg = inp['y_mlg']
    y_n = inp['y_n']
    x_n = inp['x_n']
    z_lg = inp['z_lg']

    if view == 'side':
        # Vista lateral (plano ZX, y = 0)
        z_top = max(D_f / 2 + 2.5, 6.5)
        z_bot = z_lg - 1.2
        x_left = -4.0

        _dim_segment(
            ax,
            (0.0, 0.0, z_top), (L_f, 0.0, z_top),
            (0.0, 0.0, 0.3), (L_f, 0.0, 0.3),
            r'$L_f=%.2f$ m' % L_f,
        )
        _dim_segment(
            ax,
            (0.0, 0.0, z_bot), (x_mlg, 0.0, z_bot),
            (0.0, 0.0, z_lg), (x_mlg, 0.0, z_lg),
            r'$x_{mlg}=%.2f$ m' % x_mlg,
            label_va='top',
        )
        _dim_segment(
            ax,
            (0.0, 0.0, z_bot + 0.9), (xr_w, 0.0, z_bot + 0.9),
            (0.0, 0.0, 0.0), (xr_w, 0.0, 0.0),
            r'$x_{r,w}=%.2f$ m' % xr_w,
            label_va='top',
        )
        _dim_segment(
            ax,
            (x_left, 0.0, 0.0), (x_left, 0.0, z_lg),
            (0.0, 0.0, 0.0), (0.0, 0.0, z_lg),
            r'$-z_{lg}=%.2f$ m' % (-z_lg),
            label_ha='right',
            label_offset=(-0.08, 0.0),
        )

    elif view == 'top':
        # Vista superior (plano XY, z = 0)
        y_top = b_w / 2 + 3.5
        x_right = L_f + 3.0

        _dim_segment(
            ax,
            (0.0, y_top, 0.0), (L_f, y_top, 0.0),
            (0.0, y_mlg, 0.0), (L_f, y_mlg, 0.0),
            r'$L_f=%.2f$ m' % L_f,
        )
        _dim_segment(
            ax,
            (0.0, -y_top, 0.0), (x_mlg, -y_top, 0.0),
            (0.0, -y_mlg, 0.0), (x_mlg, -y_mlg, 0.0),
            r'$x_{mlg}=%.2f$ m' % x_mlg,
            label_va='top',
        )
        _dim_segment(
            ax,
            (x_right, -b_w / 2, 0.0), (x_right, b_w / 2, 0.0),
            (L_f * 0.55, -b_w / 2, 0.0), (L_f * 0.55, b_w / 2, 0.0),
            r'$b_w=%.2f$ m' % b_w,
            label_ha='left',
            label_offset=(0.1, 0.0),
        )
        _dim_segment(
            ax,
            (x_n, -y_top - 1.2, 0.0), (x_n, -y_n, 0.0),
            (x_n + 1.5, -y_top - 1.2, 0.0), (x_n + 1.5, -y_n, 0.0),
            r'$y_n=%.2f$ m' % y_n,
            label_ha='left',
            label_offset=(0.08, 0.0),
        )

    elif view == 'front':
        # Vista frontal (plano YZ, x = 0)
        x_dim = -4.5
        z_bot = z_lg - 1.0

        _dim_segment(
            ax,
            (0.0, -b_w / 2, z_bot), (0.0, b_w / 2, z_bot),
            (0.0, -y_mlg, z_lg), (0.0, y_mlg, z_lg),
            r'$b_w=%.2f$ m' % b_w,
            label_va='top',
        )
        _dim_segment(
            ax,
            (0.0, x_dim, -D_f / 2), (0.0, x_dim, D_f / 2),
            (0.0, 0.0, -D_f / 2), (0.0, 0.0, D_f / 2),
            r'$D_f=%.2f$ m' % D_f,
            label_ha='right',
            label_offset=(-0.1, 0.0),
        )
        _dim_segment(
            ax,
            (0.0, y_n + 2.5, z_lg), (0.0, y_n, z_lg),
            (0.0, y_n + 2.5, 0.0), (0.0, y_n, 0.0),
            r'$y_{mlg}=%.2f$ m' % y_mlg,
            label_ha='left',
            label_offset=(0.08, 0.0),
        )

    return ax


def view_key_from_angles(elev, azim):
    """Mapeia elev/azim do run_airplane para o tipo de vista."""
    if elev == 90 and azim == -90:
        return 'top'
    if elev == 0 and azim == -90:
        return 'side'
    if elev == 0 and azim == 0:
        return 'front'
    return None
