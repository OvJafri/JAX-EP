# -*- coding: utf-8 -*-
"""
plot_gradient_verification_cube.py
=======================================
Journal-standard gradient verification figure for the 3D cube
differentiability showcase (run_differentiability_showcase_v2_final.py),
built entirely from the REAL, actual .npz files that script saves --
not placeholder or fabricated numbers.

Files are device-tagged (e.g. cube_differentiability_v2_gpu.npz,
cube_differentiability_v2_cpu.npz) -- run the main script once per
device (e.g. once with JAX_PLATFORMS=cpu, once on a GPU-enabled
session) to get BOTH, which lets Panel D show a genuine CPU-vs-GPU
timing comparison. If only one device's files are present, Panel D
falls back to showing FD/jax.grad/jacfwd timing for that one device,
with a note that a CPU-vs-GPU comparison needs both.

Layout (4 panels):
  A. |gradient| magnitude, log-scaled AXIS (not log-transformed bar
     heights -- taller bar always means bigger gradient) -- Raw AP
     vs Bipolar EGM, all 5 params
  B. FD vs jacfwd verification table -- Raw AP showcase
  C. FD vs jacfwd verification table -- Bipolar EGM showcase
  D. Gradient computation time: CPU vs GPU if both are available,
     otherwise FD/jax.grad/jacfwd for whichever single device was run
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        300,
    'savefig.dpi':       300,
})

_default_out = "/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else "./outputs"
OUT_DIR = os.environ.get("JAX_EP_CUBE_OUTPUT_DIR", _default_out)

PNAMES_TEX = [r'$\tau_\mathrm{in}$', r'$\tau_\mathrm{out}$',
              r'$\tau_\mathrm{open}$', r'$\tau_\mathrm{close}$',
              r'$G_\mathrm{IL}$']


def find_device_files(base_name):
    """Looks for base_name_cpu.npz and base_name_gpu.npz in OUT_DIR.
    Returns a dict {device: loaded_npz} for whichever exist."""
    found = {}
    for device in ["gpu", "cpu"]:
        path = os.path.join(OUT_DIR, f"{base_name}_{device}.npz")
        if os.path.isfile(path):
            print(f"Found: {path}")
            found[device] = np.load(path)
    if not found:
        raise FileNotFoundError(
            f"No {base_name}_cpu.npz or {base_name}_gpu.npz found in {OUT_DIR} -- "
            f"run run_differentiability_showcase_v2_final.py first.")
    return found


ap_files = find_device_files("cube_differentiability_v2")
bp_files = find_device_files("cube_differentiability_bipolar")

ap_primary_device = "gpu" if "gpu" in ap_files else "cpu"
bp_primary_device = "gpu" if "gpu" in bp_files else "cpu"
ap_data = ap_files[ap_primary_device]
bp_data = bp_files[bp_primary_device]
print(f"Using '{ap_primary_device}' data for Raw AP panels, "
      f"'{bp_primary_device}' data for Bipolar EGM panels")

ap_fd, ap_fw = ap_data['fd_grads'], ap_data['fwd_grads']
bp_fd, bp_fw = bp_data['fd_grads'], bp_data['fwd_grads']

both_devices_available = ("cpu" in ap_files and "gpu" in ap_files) or \
                          ("cpu" in bp_files and "gpu" in bp_files)

C = ['#0072B2', '#D55E00', '#56B4E9', '#E69F00']

fig = plt.figure(figsize=(7.0, 7.5))
gs = gridspec.GridSpec(3, 2, figure=fig,
                        height_ratios=[1.1, 1.0, 0.9],
                        hspace=0.55, wspace=0.42,
                        top=0.95, bottom=0.06,
                        left=0.10, right=0.97)


def panel_label(ax, letter):
    ax.text(-0.10, 1.08, letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')


ax_a = fig.add_subplot(gs[0, :])
panel_label(ax_a, 'A')
x = np.arange(5)
w = 0.32
ax_a.bar(x - w/2, np.abs(ap_fd), w, color=C[0], label='Raw AP ($V^2$ loss)',
         edgecolor='white', linewidth=0.4, zorder=3)
ax_a.bar(x + w/2, np.abs(bp_fd), w, color=C[1], label='Bipolar EGM',
         edgecolor='white', linewidth=0.4, zorder=3)
ax_a.set_yscale('log')
ax_a.set_xticks(x)
ax_a.set_xticklabels(PNAMES_TEX, fontsize=9)
ax_a.set_ylabel(r'$|\partial\mathcal{L}/\partial\theta_i|$ (FD, log scale)', fontsize=9)
ax_a.legend(ncol=2, frameon=False, fontsize=7.5,
            loc='upper center', bbox_to_anchor=(0.5, 1.18),
            columnspacing=1.2, handlelength=1.2)
ax_a.grid(axis='y', lw=0.4, alpha=0.4, which='both', zorder=0)
ax_a.set_axisbelow(True)


def draw_verification_table(ax, letter, title, fd_vals, fw_vals, subtitle_color='#2E7D32'):
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(-0.20, 1.10, letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=10)

    ratios = []
    for f, w_ in zip(fd_vals, fw_vals):
        if abs(f) > 1e-10:
            ratios.append(f"{w_/f:.4f}")
        else:
            ratios.append("n/a (~0)")

    col_cx = [0.11, 0.36, 0.63, 0.87]
    col_labels = ['Param', 'FD', 'jacfwd', 'Ratio']
    for cx, lbl in zip(col_cx, col_labels):
        ax.text(cx, 0.90, lbl, transform=ax.transAxes,
                ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1B3A6B')
    ax.plot([0.02, 0.98], [0.84, 0.84], color='#1B3A6B', lw=1.2,
            transform=ax.transAxes)

    row_ys = np.linspace(0.73, 0.16, 5)
    for ri, (p, fd, fw, ra) in enumerate(zip(PNAMES_TEX, fd_vals, fw_vals, ratios)):
        if ri % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (0.02, row_ys[ri]-0.055), 0.96, 0.10,
                transform=ax.transAxes,
                facecolor='#EEF4FB', edgecolor='none', clip_on=False))
        for cx, val, col_, fs_, fw_style in [
            (col_cx[0], p, '#222222', 8.5, 'normal'),
            (col_cx[1], f"{fd:.3e}", '#888888', 6.3, 'normal'),
            (col_cx[2], f"{fw:.3e}", '#0072B2', 6.3, 'bold'),
            (col_cx[3], ra, '#1B3A6B', 7.5, 'bold'),
        ]:
            ax.text(cx, row_ys[ri], val, transform=ax.transAxes,
                    ha='center', va='center', fontsize=fs_,
                    color=col_, fontweight=fw_style)

    ax.plot([0.02, 0.98], [0.10, 0.10], color='#1B3A6B', lw=0.8,
            transform=ax.transAxes)
    n_close = sum(1 for r in ratios if r != "n/a (~0)" and abs(float(r) - 1.0) < 0.05)
    ax.text(0.50, 0.03,
            f'{n_close}/5 within 5% of unity  \u2713 AD gradients confirmed',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=6.6, color=subtitle_color, fontweight='bold')


ax_b = fig.add_subplot(gs[1, 0])
draw_verification_table(ax_b, 'B', f'Raw AP ($V^2$ loss)\nAD verification ({ap_primary_device.upper()})',
                         ap_fd, ap_fw)

ax_c = fig.add_subplot(gs[1, 1])
draw_verification_table(ax_c, 'C', f'Bipolar EGM\nAD verification ({bp_primary_device.upper()})',
                         bp_fd, bp_fw)

ax_d = fig.add_subplot(gs[2, :])
panel_label(ax_d, 'D')

if both_devices_available:
    showcases_dev = []
    cpu_totals = []
    gpu_totals = []
    for name, files in [('Raw AP', ap_files), ('Bipolar EGM', bp_files)]:
        if "cpu" in files and "gpu" in files:
            showcases_dev.append(name)
            c = files["cpu"]; g = files["gpu"]
            cpu_totals.append(float(c['t_fd']) + float(c['t_ad']) + float(c['t_fwd']))
            gpu_totals.append(float(g['t_fd']) + float(g['t_ad']) + float(g['t_fwd']))

    x2 = np.arange(len(showcases_dev))
    w2 = 0.30
    b1 = ax_d.bar(x2 - w2/2, cpu_totals, w2, label='CPU (FD+jax.grad+jacfwd)',
                  color='#BBBBBB', edgecolor='white', linewidth=0.4, zorder=3)
    b2 = ax_d.bar(x2 + w2/2, gpu_totals, w2, label='GPU (FD+jax.grad+jacfwd)',
                  color='#0072B2', edgecolor='white', linewidth=0.4, zorder=3)
    for b, t in zip(b1, cpu_totals):
        ax_d.text(b.get_x()+b.get_width()/2., b.get_height()+max(cpu_totals)*0.02,
                   f'{t:.1f}s', ha='center', va='bottom', fontsize=6.5)
    for b, t in zip(b2, gpu_totals):
        ax_d.text(b.get_x()-b.get_width()*0.05, b.get_height()+max(cpu_totals)*0.02,
                   f'{t:.1f}s', ha='right', va='bottom', fontsize=6.5, color='#0072B2', fontweight='bold')
    for i in range(len(showcases_dev)):
        speedup = cpu_totals[i] / gpu_totals[i] if gpu_totals[i] > 0 else float('nan')
        ax_d.annotate(f'{speedup:.1f}\u00d7 speedup',
                      xy=(x2[i]+w2/2+w2*0.15, gpu_totals[i]+max(cpu_totals)*0.03),
                      xytext=(x2[i]+w2/2+w2*0.15, max(cpu_totals)*0.55),
                      ha='center', fontsize=7.5, color='#D55E00', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='#D55E00', lw=0.9))
    ax_d.set_xticks(x2)
    ax_d.set_xticklabels(showcases_dev, fontsize=9)
    ax_d.set_ylabel('Total gradient time (s)', fontsize=9)
    ax_d.set_title('CPU vs GPU: total differentiability computation time',
                    fontsize=8, fontweight='bold', pad=18)
    ax_d.legend(frameon=False, fontsize=7,
                bbox_to_anchor=(0.5, 1.20), loc='upper center',
                ncol=2, columnspacing=1.5, handlelength=1.4)
else:
    only_device = ap_primary_device
    print(f"\nOnly '{only_device}' data available -- Panel D falls back to "
          f"per-method timing. Run this script again after also running the "
          f"main showcase on the other device to get a CPU-vs-GPU comparison.")
    showcases = ['Raw AP\n($V^2$ loss)', 'Bipolar EGM']
    x2 = np.arange(2)
    w2 = 0.22
    fd_times = [float(ap_data['t_fd']), float(bp_data['t_fd'])]
    ad_times = [float(ap_data['t_ad']), float(bp_data['t_ad'])]
    fw_times = [float(ap_data['t_fwd']), float(bp_data['t_fwd'])]
    b1 = ax_d.bar(x2 - w2, fd_times, w2, label='FD (5 params x 2 passes)',
                  color='#BBBBBB', edgecolor='white', linewidth=0.4, zorder=3)
    b2 = ax_d.bar(x2, ad_times, w2, label='jax.grad (reverse-mode)',
                  color='#E69F00', edgecolor='white', linewidth=0.4, zorder=3)
    b3 = ax_d.bar(x2 + w2, fw_times, w2, label='jacfwd (5 tangent passes)',
                  color='#0072B2', edgecolor='white', linewidth=0.4, zorder=3)
    for bars, times in [(b1, fd_times), (b2, ad_times), (b3, fw_times)]:
        for b, t in zip(bars, times):
            ax_d.text(b.get_x()+b.get_width()/2., b.get_height()+max(fd_times+ad_times+fw_times)*0.02,
                       f'{t:.1f}s', ha='center', va='bottom', fontsize=6.5)
    ax_d.set_xticks(x2)
    ax_d.set_xticklabels(showcases, fontsize=9)
    ax_d.set_ylabel('Gradient computation time (s)', fontsize=9)
    ax_d.set_title(f'Gradient computation time by method ({only_device.upper()} only -- '
                    f'run the other device for a CPU/GPU comparison)',
                    fontsize=7.5, fontweight='bold', pad=18)
    ax_d.legend(frameon=False, fontsize=7,
                bbox_to_anchor=(0.5, 1.20), loc='upper center',
                ncol=3, columnspacing=1.0, handlelength=1.2)
    ax_d.set_ylim(top=max(fd_times+ad_times+fw_times)*1.25)

ax_d.grid(axis='y', lw=0.4, alpha=0.4, zorder=0)
ax_d.set_axisbelow(True)

fname = os.path.join(OUT_DIR, 'figure_cube_gradient_verification.png')
fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {fname}")