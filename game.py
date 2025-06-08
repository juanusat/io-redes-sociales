import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

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

def leader_strategy(users, subcats, pub_dur, k=2, epsilon=1):
    all_subs = [sc for lst in subcats.values() for sc in lst]
    if np.random.rand() < epsilon:
        chosen = np.random.choice(all_subs, size=k, replace=False).tolist()
    else:
        expected = {}
        for lst in subcats.values():
            for sc in lst:
                payoff = sum(
                    (pub_dur * u['aff'].get(sc, 0)) - 10 * int(pub_dur * u['aff'].get(sc, 0) < 5)
                    for u in users if u['connected']
                )
                expected[sc] = payoff
        chosen = sorted(expected, key=expected.get, reverse=True)[:k]
    return [{'sub': sc, 'dur': pub_dur} for sc in chosen]

def format_hms(seconds):
    return str(timedelta(seconds=int(seconds)))

def simulate(users, subcats, dur, report_interval):
    stats = {'attention': {sc: 0 for lst in subcats.values() for sc in lst}, 'count': 0}
    sim = 0
    step = 30
    total_steps = int(dur / step)
    real_start = time.time()
    next_report = report_interval
    report_num = 1
    step_count = 0

    while sim < dur:
        pubs = leader_strategy(users, subcats, pub_dur=step, k=2)
        for u in users:
            if not u['connected']:
                continue
            for pub in pubs:
                t, left = evaluate_attention(u, pub['dur'], pub['sub'])
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

def save_results(stats, cat_dist, subcats, start, end, args):
    labels = list(subcats.keys())
    real_vals = [cat_dist.get(c, 0) for c in labels]
    total_learned = sum(sum(stats['attention'][sc] for sc in subcats[c]) for c in labels) or 1
    learned_vals = [(sum(stats['attention'][sc] for sc in subcats[c]) / total_learned * 100) for c in labels]

    diffs = [a - r for r, a in zip(real_vals, learned_vals)]
    mae = sum(abs(d) for d in diffs) / len(diffs)
    rmse = math.sqrt(sum(d**2 for d in diffs) / len(diffs))

    if args.c:
        print("\nResumen final de preferencias:")
        print(f"{'Categoría':<15} {'Real (%)':>10} {'Aprendida (%)':>15} {'Diferencia':>12}")
        for l, r, a in zip(labels, real_vals, learned_vals):
            diff = a - r
            print(f"{l:<15} {r:10.2f} {a:15.2f} {diff:12.2f}")
        print(f"\nError absoluto medio (MAE): {mae:.2f}")
        print(f"Error cuadrático medio (RMSE): {rmse:.2f}")

    out = Path('results')
    out.mkdir(exist_ok=True)
    idx = len(list(out.glob('prueba*'))) + 1
    dst = out / f'prueba{idx:03d}'
    dst.mkdir()
    print("")
    print(f"Resultados guardados en: {dst}")
    plt.figure(); plt.bar(labels, real_vals); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(dst/'real_pref.jpeg')
    plt.figure(); plt.bar(labels, learned_vals); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(dst/'learned_pref.jpeg')
    info = {
        'config': vars(args),
        'inicio': start.isoformat(),
        'fin': end.isoformat(),
        'dur_sim_fict_s': (end-start).total_seconds(),
        'total_pubs': stats['count'],
        'avg_attention': sum(stats['attention'].values())/stats['count'] if stats['count'] else 0,
        'att_sub': stats['attention'],
        'att_cat_pct': dict(zip(labels, learned_vals))
    }
    with open(dst/'informe.json', 'w') as f:
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

    stats, total_steps = simulate(users, subcats, dur, args.r)
    end = datetime.now()
    print(f"Fin simulación: {fecha_legible(end)}")
    print(f"Tiempo real total: {int((time.time() - time.mktime(start.timetuple())))}s")
    save_results(stats, cat_dist, subcats, start, end, args)

if __name__ == '__main__':
    main()