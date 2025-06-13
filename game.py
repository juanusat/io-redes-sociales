import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

def fecha_legible(dt):
    return dt.strftime('%Y/%m/%d %I:%M:%S.') + f"{int(dt.microsecond/1000):03d} " + dt.strftime('%p').lower()

def parse_time(s):
    unit = s[-1]
    v = int(s[:-1])
    return {'s': v, 'm': v*60, 'h': v*3600, 'd': v*86400}.get(unit,
        ValueError(f"Unidad no reconocida: {unit}"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='Ruta CSV con población por categoría')
    parser.add_argument('-d', default='30m', help='Duración total de simulación (tiempo ficticio)')
    parser.add_argument('-m', type=int, default=100, help='Número de usuarios a simular')
    parser.add_argument('-r', type=int, default=0, help='Intervalo de reporte en segundos reales')
    parser.add_argument('-c', action='store_true', help='Mostrar en consola preferencias reales y aprendidas')
    parser.add_argument('-g', action='store_true', help='Mostrar gráfica en ventana además de guardarla')
    parser.add_argument('-e', type=float, default=0.1, help='Valor de epsilon para estrategia líder (default 0.1)')
    return parser.parse_args()

def load_categories(path):
    df = pd.read_csv(path)
    return df.groupby('categoria')['subcategoria'].apply(list).to_dict()

def load_population(path):
    df = pd.read_csv(path)
    return df.set_index('categoria')['porcentaje'].to_dict()

def init_users(n, cat_dist, subcats):
    users = []
    for i in range(n):
        base = {c: max(0, np.random.normal(cat_dist.get(c, 0), 5)) for c in subcats}
        total = sum(base.values()) or 1
        cat_aff = {c: v/total for c, v in base.items()}
        sub_aff = {}
        for c, lst in subcats.items():
            for sc in lst:
                sub_aff[sc] = max(0, cat_aff[c] + np.random.normal(0, 0.1))
        tot_sc = sum(sub_aff.values()) or 1
        sub_aff = {sc: v/tot_sc for sc, v in sub_aff.items()}
        users.append({'id': i, 'aff': sub_aff, 'connected': True, 'unint': 0})
    return users

def evaluate_attention(user, pub_dur, sub):
    t = pub_dur * user['aff'].get(sub, 0)
    left = False
    if t < 5:
        user['unint'] += 1
        if user['unint'] >= 3:
            user['connected'] = False
            user['unint'] = 0
            left = True
    else:
        user['unint'] = 0
    return t, left


def handle_recovery(users):
    for u in users:
        if not u['connected']:
            top = max(u['aff'], key=u['aff'].get)
            if u['aff'][top] > np.random.rand():
                u['connected'] = True

history = defaultdict(list)

def leader_strategy_history(history, subcats, pub_dur, k=2, epsilon=0.1):
    all_subs = [sc for lst in subcats.values() for sc in lst]

    # Calcular promedio de atención para cada subcategoría, si no hay datos usar 0
    avg_att = {
        sc: np.mean(history[sc]) if history[sc] else 1
        for sc in all_subs
    }

    # Probabilidad de exploración: elegir aleatoriamente k subcategorías
    if np.random.rand() < epsilon:
        chosen = np.random.choice(all_subs, size=k, replace=False).tolist()
    else:
        # Explotación: elegir las k subcategorías con mayor atención promedio
        chosen = sorted(avg_att, key=avg_att.get, reverse=True)[:k]

    return [{'sub': sc, 'dur': pub_dur} for sc in chosen]

def format_hms(seconds):
    return str(timedelta(seconds=int(seconds)))

def simulate(users, subcats, dur, report_interval, epsilon=0.1):
    stats = {'attention': {sc: 0 for lst in subcats.values() for sc in lst}, 'count': 0}
    sim = 0
    step = 30
    total_steps = int(dur / step)
    real_start = time.time()
    next_report = report_interval
    report_num = 1
    step_count = 0

    # Tiempo previo para mostrar 2 publicaciones de cada subcategoría sin estrategia (epsilon=1)
    all_subs = [sc for lst in subcats.values() for sc in lst]
    pre_steps = 2  # 2 publicaciones por subcategoría
    pre_dur = pre_steps * step
    print(f"Tiempo previo para exploración sin estrategia: {pre_dur}s")

    for _ in range(pre_steps):
        pubs = [{'sub': sc, 'dur': step} for sc in all_subs]
        for u in users:
            if not u['connected']:
                continue
            for pub in pubs:
                t, left = evaluate_attention(u, pub['dur'], pub['sub'])
                history[pub['sub']].append(t)
                stats['attention'][pub['sub']] += t
                stats['count'] += 1
                if left:
                    break
        handle_recovery(users)
        sim += step
        step_count += 1

    # Simulación principal con estrategia líder y epsilon dado
    while sim < dur:
        pubs = leader_strategy_history(history, subcats, pub_dur=step, k=2, epsilon=epsilon)
        for u in users:
            if not u['connected']:
                continue
            for pub in pubs:
                t, left = evaluate_attention(u, pub['dur'], pub['sub'])
                history[pub['sub']].append(t)
                stats['attention'][pub['sub']] += t
                stats['count'] += 1
                if left:
                    break
        handle_recovery(users)
        sim += step
        step_count += 1

        if report_interval and (time.time() - real_start) >= next_report:
            elapsed = time.time() - real_start
            progress_pct = step_count / total_steps * 100
            avg_time_per_step = elapsed / step_count
            steps_left = total_steps - step_count
            est_remain = int(steps_left * avg_time_per_step)
            total_reports = int(dur / step / (report_interval / step)) if report_interval >= step else report_interval
            print(
                f"Reporte {report_num}: Paso {step_count}/{total_steps} ({progress_pct:5.2f}%) | "
                f"Real transcurrido: {int(elapsed)}s | "
                f"Real restante estimado: {est_remain}s"
            )
            report_num += 1
            next_report += report_interval

    return stats, total_steps

def save_results(stats, cat_dist, subcats, start, end, args, users):
    labels = list(subcats.keys())
    sample_vals = []
    for c in labels:
        avg_aff = sum(
            sum(u['aff'][sc] for sc in subcats[c])
            for u in users
        ) / len(users)
        sample_vals.append(avg_aff * 100)
    real_vals = [cat_dist.get(c, 0) for c in labels]
    total_learned = sum(sum(stats['attention'][sc] for sc in subcats[c]) for c in labels) or 1
    learned_vals = [
        (sum(stats['attention'][sc] for sc in subcats[c]) / total_learned * 100)
        for c in labels
    ]

    # Calcular porcentaje de publicaciones mostradas por categoría
    total_hist = sum(len(history[sc]) for lst in subcats.values() for sc in lst) or 1
    hist_per_cat = []
    for c in labels:
        count_cat = sum(len(history[sc]) for sc in subcats[c])
        hist_per_cat.append((count_cat / total_hist) * 100)

    diffs = [a - m for m, a in zip(sample_vals, learned_vals)]
    mae = sum(abs(d) for d in diffs) / len(diffs)
    rmse = math.sqrt(sum(d**2 for d in diffs) / len(diffs))

    if args.c:
        print("\nResumen final de preferencias:")
        print(f"{'Categoría':<15} {'Real (%)':>10} {'Muestra (%)':>12} {'Aprendida (%)':>15} {'Diferencia':>12} {'Historial (%)':>15}")
        for l, r, m, a, d, h in zip(labels, real_vals, sample_vals, learned_vals, diffs, hist_per_cat):
            print(f"{l:<15} {r:10.2f} {m:12.2f} {a:15.2f} {d:12.2f} {h:15.2f}")
        print(f"\nError absoluto medio (MAE): {mae:.2f}")
        print(f"Error cuadrático medio (RMSE): {rmse:.2f}")


    out = Path('results')
    out.mkdir(exist_ok=True)
    idx = len(list(out.glob('prueba*'))) + 1
    dst = out / f'prueba{idx:03d}'
    dst.mkdir()
    print(f"\nResultados guardados en: {dst}")

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    bars1 = ax.bar(x - 1.5*width, real_vals,   width, label='Real')
    bars2 = ax.bar(x - 0.5*width, sample_vals, width, label='Muestra')
    bars3 = ax.bar(x + 0.5*width, learned_vals, width, label='Aprendida')
    bars4 = ax.bar(x + 1.5*width, hist_per_cat, width, label='Historial')

    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Porcentaje (%)')
    all_vals = real_vals + sample_vals + learned_vals + hist_per_cat
    ymin, ymax = min(all_vals), max(all_vals)
    delta = (ymax - ymin) * 0.1 if ymax > ymin else ymax * 0.1
    ax.set_ylim(ymin - delta, ymax + delta)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    out_path = dst / 'comparativa_pref.jpg'
    fig.savefig(out_path, dpi=100)
    if args.g:
        plt.show()
    plt.close(fig)

    info = {
        'config': vars(args),
        'inicio': start.isoformat(),
        'fin': end.isoformat(),
        'dur_sim_fict_s': (end - start).total_seconds(),
        'total_pubs': stats['count'],
        'avg_attention': sum(stats['attention'].values()) / stats['count'] if stats['count'] else 0,
        'att_sub': stats['attention'],
        'att_cat_pct': dict(zip(labels, learned_vals))
    }
    with open(dst / 'informe.json', 'w') as f:
        json.dump(info, f, indent=2)

def main():
    args = parse_args()
    dur = parse_time(args.d)
    start = datetime.now()
    print(f"Inicio simulación: {fecha_legible(start)}")
    print(f"Duración (ficticia): {args.d} ({dur}s) | Usuarios: {args.m} | Reporte cada: {args.r}s")

    subcats = load_categories(Path('data') / 'lista-categorias.csv')
    cat_dist = load_population(Path(args.p))
    users = init_users(args.m, cat_dist, subcats)

    stats, total_steps = simulate(users, subcats, dur, args.r, epsilon=args.e)
    end = datetime.now()
    print(f"Fin simulación: {fecha_legible(end)}")
    print(f"Tiempo real total: {int((time.time() - time.mktime(start.timetuple())))}s")

    save_results(stats, cat_dist, subcats, start, end, args, users)

if __name__ == '__main__':
    main()
