#!/usr/bin/env python3
import argparse
import numpy as np

# ---------------- utils ----------------
def reshape_if_flat(a, H, W):
    if a.ndim == 1:
        if a.size != H * W:
            raise ValueError(f"Cannot reshape {a.size} elements into ({H},{W})")
        a = a.reshape(H, W)
    elif a.ndim != 2:
        raise ValueError("Expected 1D or 2D array")
    return a

def make_xy(H, W, sx, sy):
    y = np.arange(H) * sy
    x = np.arange(W) * sx
    X, Y = np.meshgrid(x, y)
    return X, Y

def downsample(*arrays, step=4):
    out = []
    for a in arrays:
        out.append(a[::step, ::step] if a is not None else None)
    return out if len(out) > 1 else out[0]

def export_ply(xyz, rgb, out_ply):
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    if rgb is not None:
        rgb = rgb[mask]
    with open(out_ply, "w") as f:
        n = xyz.shape[0]
        has_color = rgb is not None
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_color:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z) in xyz:
                f.write(f"{x} {y} {z}\n")
    print(f"Saved PLY: {out_ply}")

def try_plotly(X, Y, Z_vis, lumi=None, out_html="height3d.html"):
    try:
        import plotly.graph_objects as go
        if lumi is not None:
            lmin, lmax = np.nanmin(lumi), np.nanmax(lumi)
            l = (lumi - lmin) / max(1e-12, (lmax - lmin))
            surface = go.Surface(x=X, y=Y, z=Z_vis, surfacecolor=l, colorbar=dict(title="Lumi"))
        else:
            surface = go.Surface(x=X, y=Y, z=Z_vis, colorbar=dict(title="Height (vis)"))
        fig = go.Figure(data=[surface])
        fig.update_layout(scene_aspectmode="data", title="Interactive 3D")
        fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)
        print(f"Saved interactive HTML: {out_html}")
        return True
    except Exception as e:
        print(f"[plotly] fallback to matplotlib ({e})")
        return False

def try_open3d(X, Y, Z, lumi=None):
    try:
        import open3d as o3d
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        m = np.isfinite(pts).all(axis=1)
        pts = pts[m]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if lumi is not None:
            l = lumi.ravel()[m]
            l = (l - np.nanmin(l)) / max(1e-12, (np.nanmax(l) - np.nanmin(l)))
            lrgb = np.stack([l, l, l], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(lrgb.astype(np.float64))
        o3d.visualization.draw_geometries([pcd])
        return True
    except Exception as e:
        print(f"[open3d] fallback to matplotlib ({e})")
        return False

def do_matplotlib(X, Y, Z_vis, lumi=None, out_png="height3d.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    from matplotlib import cm

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 用 NaN 表示不绘制的点
    Zm = ma.masked_invalid(Z_vis)

    # 准备 colormap，并将 mask 区域设为透明
    if lumi is not None:
        cmap = cm.get_cmap("gray").copy()
    else:
        cmap = cm.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)  # 关键：mask/NaN 区域完全透明

    if lumi is not None:
        # 用 lumi 灰度着色，但透明由 Z 的 mask 决定
        L = lumi
        lmin, lmax = np.nanmin(L), np.nanmax(L)
        ln = (L - lmin) / max(1e-12, (lmax - lmin))
        ln_masked = ma.array(ln, mask=ma.getmaskarray(Zm))

        # 生成 RGBA 普通 ndarray（坏值透明）
        fc_full = cmap(ln_masked)
        fc_full = np.asarray(fc_full)  # 确保不是 masked_array

        # facecolors 与 surface 网格对齐（(H-1)*(W-1) 个面）
        H, W = Zm.shape
        Xd = X[:H-1, :W-1]
        Yd = Y[:H-1, :W-1]
        Zd = Zm[:H-1, :W-1]
        fc = fc_full[:H-1, :W-1, :]

        ax.plot_surface(
            Xd, Yd, Zd,
            rstride=1, cstride=1, linewidth=0, antialiased=True,
            facecolors=fc, cmap=None,
            shade=False  # 避免二次着色导致 RGBA 解析异常
        )
    else:
        # 直接用 Z 与 colormap（mask 区域透明）
        ax.plot_surface(
            X, Y, Zm,
            rstride=1, cstride=1, linewidth=0, antialiased=True,
            facecolors=None, cmap=cmap,
            shade=False
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm, visual)")
    ax.set_xlim([0,320])
    ax.set_ylim([0,320])
    def _ptp(a): return np.nanmax(a) - np.nanmin(a)
    ax.set_box_aspect((_ptp(X), _ptp(Y), np.nanmax(Z_vis) - np.nanmin(Z_vis)))
    ax.view_init(elev=60, azim=-60)

    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 200000
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved static 3D PNG: {out_png}")

def mirror_axis(a, axis):
    return np.concatenate([np.flip(a, axis=axis), a], axis=axis)

def auto_majority_value(a):
    flat = a.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    sample = flat if flat.size <= 2_000_000 else np.random.choice(flat, 2_000_000, replace=False)
    vals, counts = np.unique(sample, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx], counts[idx] / sample.size

def apply_masking(Z, L, args):
    mask = np.zeros_like(Z, dtype=bool)

    # —— Z=0 作为基准面：默认开启，不绘制（透明） ——
    if args.zero_as_base:
        mask |= np.isclose(Z, 0.0, rtol=0.0, atol=1e-12)

    if args.mask_value is not None:
        mask |= np.isclose(Z, args.mask_value, rtol=0.0, atol=args.mask_eps)
    if args.auto_mask_majority:
        res = auto_majority_value(Z)
        if res is not None:
            maj, frac = res
            if frac >= args.majority_min_frac:
                mask |= np.isclose(Z, maj, rtol=0.0, atol=args.mask_eps)
                print(f"[mask] auto majority value={maj} (~{frac:.1%}) masked (eps={args.mask_eps})")
    if args.mask_below is not None:
        mask |= (Z < args.mask_below)
    if args.mask_above is not None:
        mask |= (Z > args.mask_above)

    Zm = np.where(mask, np.nan, Z)
    Lm = None if L is None else np.where(mask, np.nan, L)

    if args.clip_low_perc is not None or args.clip_high_perc is not None:
        p_lo = args.clip_low_perc if args.clip_low_perc is not None else 0.0
        p_hi = args.clip_high_perc if args.clip_high_perc is not None else 100.0
        finite = np.isfinite(Zm)
        if finite.any():
            lo = np.percentile(Zm[finite], p_lo)
            hi = np.percentile(Zm[finite], p_hi)
            Zm = np.clip(Zm, lo, hi)
    return Zm, Lm

def maybe_transpose_for_y_longer(Z, L, sx, sy, ensure_y_longer=False, force_transpose=False, swap_xy=False):
    if force_transpose:
        Z = Z.T
        L = None if L is None else L.T
        sx, sy = sy, sx
    if swap_xy and not force_transpose:
        sx, sy = sy, sx
    H, W = Z.shape
    if ensure_y_longer and (W * sx > H * sy):
        Z = Z.T
        L = None if L is None else L.T
        sx, sy = sy, sx
    return Z, L, sx, sy

def auto_zvis_for_aspect(X, Y, Z, target_ratio=1.0):
    ptpX = np.nanmax(X) - np.nanmin(X)
    ptpY = np.nanmax(Y) - np.nanmin(Y)
    ptpZ = np.nanmax(Z) - np.nanmin(Z)
    if ptpZ <= 0:
        return 1.0
    targetZ = target_ratio * 0.5 * (ptpX + ptpY)
    return max(1e-6, targetZ / ptpZ)

# -------- side-by-side height & lumi figure --------
def save_side_by_side_height_lumi(X, Y, Z_vis, L, out_png="height_lumi_cmp.png",
                                  elev=60, azim=-60):
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import matplotlib.cm as cm

    Zm = ma.masked_invalid(Z_vis)
    Lm = None if L is None else ma.masked_array(L, mask=ma.getmaskarray(Zm))

    fig = plt.figure(figsize=(14, 6))

    # 左：按高度着色
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    cmap1 = cm.get_cmap("viridis").copy()
    cmap1.set_bad(alpha=0.0)
    s1 = ax1.plot_surface(X, Y, Zm, rstride=1, cstride=1, linewidth=0, antialiased=True,
                          cmap=cmap1, shade=False)
    ax1.set_title("Height (colored by Z)")
    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)"); ax1.set_zlabel("Z (mm)")
    def _ptp(a): return np.nanmax(a) - np.nanmin(a)
    ax1.set_box_aspect((_ptp(X), _ptp(Y), np.nanmax(Z_vis) - np.nanmin(Z_vis)))
    ax1.view_init(elev=elev, azim=azim)
    cb1 = fig.colorbar(s1, ax=ax1, shrink=0.7, pad=0.1)
    cb1.set_label("Z (mm)")

    # 右：按 lumi 灰度着色
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if Lm is not None:
        cmap2 = cm.get_cmap("gray").copy()
        cmap2.set_bad(alpha=0.0)
        lmin, lmax = np.nanmin(Lm), np.nanmax(Lm)
        ln = (Lm - lmin) / max(1e-12, (lmax - lmin))
        ln_masked = ma.array(ln, mask=ma.getmaskarray(Zm))
        fc_full = cmap2(ln_masked)
        fc_full = np.asarray(fc_full)

        H, W = Zm.shape
        Xd = X[:H-1, :W-1]
        Yd = Y[:H-1, :W-1]
        Zd = Zm[:H-1, :W-1]
        fc = fc_full[:H-1, :W-1, :]

        ax2.plot_surface(Xd, Yd, Zd, rstride=1, cstride=1, linewidth=0,
                         antialiased=True, facecolors=fc, cmap=None, shade=False)
        import matplotlib.cm as cm_
        mappable = cm_.ScalarMappable(cmap="gray")
        mappable.set_array(Lm)
        cb2 = fig.colorbar(mappable, ax=ax2, shrink=0.7, pad=0.1)
        cb2.set_label("Lumi (a.u.)")
    else:
        cmap2 = cm.get_cmap("viridis").copy()
        cmap2.set_bad(alpha=0.0)
        s2 = ax2.plot_surface(X, Y, Zm, rstride=1, cstride=1, linewidth=0,
                              antialiased=True, cmap=cmap2, shade=False)
        cb2 = fig.colorbar(s2, ax=ax2, shrink=0.7, pad=0.1)
        cb2.set_label("Z (mm)")

    ax2.set_title("Height geometry (colored by Lumi)")
    ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Y (mm)"); ax2.set_zlabel("Z (mm)")
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_box_aspect((_ptp(X), _ptp(Y), np.nanmax(Z_vis) - np.nanmin(Z_vis)))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved side-by-side PNG: {out_png}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="LJ-X8000A 3D viewer (mm units, with side-by-side Lumi)")
    ap.add_argument("--height_npy", default="z_data.npy")
    ap.add_argument("--lumi_npy", default="lumi_data.npy")
    ap.add_argument("--rows", type=int, default=1600)
    ap.add_argument("--cols", type=int, default=3200)
    # 物理像素间距默认：X=320mm/3199, Y=160mm/1599
    ap.add_argument("--sx", type=float, default=320.0/3199.0, help="mm per pixel in X")
    ap.add_argument("--sy", type=float, default=160.0/1599.0, help="mm per pixel in Y")
    ap.add_argument("--step", type=int, default=4, help="downsample step for 3D")

    # Z 缩放分离：数据尺度 vs 可视化尺度
    ap.add_argument("--zscale", type=float, default=0.001, help="DATA scale for Z (e.g., µm->mm = 0.001)")
    ap.add_argument("--zvis",   type=float, default=1.0,   help="VISUAL scale for Z only (does not affect data/export)")
    ap.add_argument("--auto-zvis", type=float, default=None,
                    help="Auto compute zvis so that Z span ~= XY avg span * this factor (e.g., 1.0)")

    # 形状/轴
    ap.add_argument("--transpose", action="store_true", help="transpose Z/L before plotting")
    ap.add_argument("--swap-xy", action="store_true", help="swap sx/sy without transposing data")
    ap.add_argument("--ensure-y-longer", action="store_true", help="auto transpose to make Y the longer side")

    # 从 quarter 重建 full
    ap.add_argument("--mirror-x", action="store_true", help="mirror along X (width)")
    ap.add_argument("--mirror-y", action="store_true", help="mirror along Y (height)")

    # 背景屏蔽与对比度
    ap.add_argument("--zero-as-base", dest="zero_as_base", action="store_true", default=True,
                    help="treat Z=0 as base plane (transparent/no color) [default ON]")
    ap.add_argument("--no-zero-as-base", dest="zero_as_base", action="store_false",
                    help="disable treating Z=0 as base plane")
    ap.add_argument("--mask-value", type=float, default=None, help="explicit Z value to mask (e.g., 0)")
    ap.add_argument("--mask-eps", type=float, default=1e-6, help="tolerance for mask")
    ap.add_argument("--auto-mask-majority", action="store_true", help="mask the majority value (likely background)")
    ap.add_argument("--majority-min-frac", type=float, default=0.4, help="fraction threshold for majority masking")
    ap.add_argument("--mask-below", type=float, default=None, help="mask Z < value")
    ap.add_argument("--mask-above", type=float, default=None, help="mask Z > value")
    ap.add_argument("--clip-low-perc", type=float, default=1.0, help="low percentile clip for Z (visual)")
    ap.add_argument("--clip-high-perc", type=float, default=99.0, help="high percentile clip for Z (visual)")

    # 输出/查看器与合图
    ap.add_argument("--viewer", choices=["auto", "plotly", "open3d", "mpl"], default="auto")
    ap.add_argument("--out_prefix", default="height3d")
    ap.add_argument("--export_ply", action="store_true", help="export downsampled point cloud PLY")
    ap.add_argument("--combine", action="store_true",
                    help="save a side-by-side comparison (left: colored by Z, right: colored by Lumi)")

    args = ap.parse_args()

    # 载入原始 Z/L
    H0, W0 = args.rows, args.cols
    Z_raw = np.load(args.height_npy, allow_pickle=False)
    print(f"[debug] Z raw shape={Z_raw.shape}, dtype={Z_raw.dtype}, "
          f"min={np.nanmin(Z_raw)}, max={np.nanmax(Z_raw)}")

    Z = reshape_if_flat(Z_raw, H0, W0).astype(np.float32)
    L = None
    if args.lumi_npy:
        L_raw = np.load(args.lumi_npy, allow_pickle=False)
        print(f"[debug] L raw shape={L_raw.shape}, dtype={L_raw.dtype}, "
              f"min={np.nanmin(L_raw)}, max={np.nanmax(L_raw)}")
        L = reshape_if_flat(L_raw, H0, W0).astype(np.float32)

    # quarter -> full（先做）
    if args.mirror_x:
        Z = mirror_axis(Z, axis=1)
        if L is not None: L = mirror_axis(L, axis=1)
    if args.mirror_y:
        Z = mirror_axis(Z, axis=0)
        if L is not None: L = mirror_axis(L, axis=0)

    # 轴/比例调整
    Z, L, args.sx, args.sy = maybe_transpose_for_y_longer(
        Z, L, args.sx, args.sy,
        ensure_y_longer=args.ensure_y_longer,
        force_transpose=args.transpose,
        swap_xy=args.swap_xy
    )

    # —— 数据尺度：把 Z 转为 mm（默认 µm->mm = 0.001）——
    if args.zscale != 1.0:
        Z = Z * args.zscale
        print(f"[debug] Z after DATA scale (zscale={args.zscale}): "
              f"min={np.nanmin(Z)}, max={np.nanmax(Z)}")
    else:
        print(f"[debug] Z after DATA scale (zscale=1): "
              f"min={np.nanmin(Z)}, max={np.nanmax(Z)}")

    # 可视化前的屏蔽/裁剪（这里会把 Z==0 变为 NaN，从而不绘制）
    Z_disp, L_disp = apply_masking(Z, L, args)

    # 自动可视化 z 缩放
    X_tmp, Y_tmp = make_xy(*Z_disp.shape, args.sx, args.sy)
    if args.auto_zvis is not None:
        sugg = auto_zvis_for_aspect(X_tmp, Y_tmp, Z_disp, target_ratio=args.auto_zvis)
        print(f"[debug] auto zvis suggestion = {sugg}")
        args.zvis = sugg

    # 可视化尺度
    Z_vis = Z_disp * args.zvis
    print(f"[debug] Z for VIS (zvis={args.zvis}): "
          f"min={np.nanmin(Z_vis)}, max={np.nanmax(Z_vis)}")

    # 坐标网格（mm）
    X, Y = X_tmp, Y_tmp

    # 降采样
    Xd, Yd, Z_vis_d, Ld = downsample(X, Y, Z_vis, L_disp, step=args.step)
    _,  _,  Z_data_d, _  = downsample(X, Y, Z_disp, L_disp, step=args.step)

    # 导出 PLY（数据尺度）
    if args.export_ply:
        xyz = np.stack([Xd.ravel(), Yd.ravel(), Z_data_d.ravel()], axis=1)
        rgb = None
        if Ld is not None:
            ln = (Ld - np.nanmin(Ld)) / max(1e-12, (np.nanmax(Ld) - np.nanmin(Ld)))
            rgb = (np.stack([ln.ravel(), ln.ravel(), ln.ravel()], axis=1) * 255).astype(np.uint8)
        export_ply(xyz, rgb, f"{args.out_prefix}.ply")

    # 并排对比图（height vs lumi）
    if args.combine:
        save_side_by_side_height_lumi(
            Xd, Yd, Z_vis_d, Ld,
            out_png=f"{args.out_prefix}_combo.png",
            elev=60, azim=-60
        )

    # 选择查看器
    used = False
    if args.viewer in ("auto", "plotly"):
        used = try_plotly(Xd, Yd, Z_vis_d, lumi=Ld, out_html=f"{args.out_prefix}.html")
        if args.viewer == "plotly":
            return
    if not used and args.viewer in ("auto", "open3d"):
        used = try_open3d(Xd, Yd, Z_data_d, lumi=Ld)
        if args.viewer == "open3d":
            return

    # fallback: matplotlib（可视化尺度）
    do_matplotlib(Xd, Yd, Z_vis_d, lumi=Ld, out_png=f"{args.out_prefix}.png")

if __name__ == "__main__":
    main()
