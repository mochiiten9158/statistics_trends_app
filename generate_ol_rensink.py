import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import truncnorm


def generate_harrison_ordered_line(corr=0.0, seed=None, out_dir="assets/harrison_ordered_positive_100", count=0):
    rng = np.random.default_rng(seed)
    n = 100

    def truncated_normal(rng, n):
        a, b = -2, 2
        return truncnorm.rvs(a, b, size=n, random_state=rng)

    x = truncated_normal(rng, n)
    y = truncated_normal(rng, n)

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    r = corr
    
    if abs(abs(r) - 1.0) < 1e-9:
        y_new = x if r > 0 else (-x)
    else:
        sign = 1.0 if r >= 0 else -1.0
        y_work = y * sign
        r_work = abs(r)

        rz = np.corrcoef(x, y_work)[0, 1]

        denom = (rz - 1) * (2 * r_work**2 + rz - 1)
        if abs(denom) < 1e-12:
            y_new = r_work * x + np.sqrt(max(0, 1 - r_work**2)) * y_work
        else:
            discriminant = r_work**2 * (rz**2 - 1) * (r_work**2 - 1)
            if discriminant < 0:
                discriminant = 0.0
            lam = ((rz - 1) * (r_work**2 + rz) + np.sqrt(discriminant)) / denom
            y_new = (lam * x + (1 - lam) * y_work) / np.sqrt(lam**2 + (1 - lam)**2)

        y_new = y_new * sign            

    current_std = x.std()

    x_final = (x / current_std) * 0.2 + 0.5
    y_final = (y_new / current_std) * 0.2 + 0.5


    sort_indices = np.argsort(x_final)
    x_ordered = x_final[sort_indices]
    y_ordered = y_final[sort_indices]
    
    t = np.linspace(0, 1, n)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{count}_{r:.4f}.png"

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.plot(
        t, 
        x_ordered, 
        color="black", 
        linestyle="--", 
        linewidth=1.2, 
        zorder=2
    )
    
    ax.plot(
        t, 
        y_ordered, 
        color="black", 
        linestyle="-", 
        linewidth=1.2, 
        zorder=1
    )

    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout(pad=0.1)
    fig.savefig(filename, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    # print(f"Saved: {filename}  "
    #       f"(target r={r:.4f}, "
    #       f"achieved r={np.corrcoef(x_final, y_final)[0,1]:.4f})")

    return filename


def main():
    seed = 1
    corrs_positive = [round(i * 0.0050, 4) for i in range(0, 201)]
    corrs_negative = [-round(i * 0.0050, 4) for i in range(0, 201)]

    for r in corrs_positive:
        generate_harrison_ordered_line(
            corr=r, 
            seed=seed,
            out_dir="assets/harrison_ordered_positive_100_testing",
            count=0
        )
        seed += 1

    seed = 5000
    for r in corrs_negative:
        generate_harrison_ordered_line(
            corr=r, 
            seed=seed,
            out_dir="assets/harrison_ordered_negative_100_testing",
            count=0
        )
        seed += 1


if __name__ == "__main__":
    main()