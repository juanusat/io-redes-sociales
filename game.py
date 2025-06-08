import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from pathlib import Path

def parse_time(s):
    unit = s[-1]
    v = int(s[:-1])
    if unit == 's':
        return v
    elif unit == 'm':
        return v * 60
    elif unit == 'h':
        return v * 3600
    elif unit == 'd':
        return v * 86400
    else:
        raise ValueError(f"Unidad de tiempo no reconocida: {unit}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store_true')
    parser.add_argument('-p', required=True, help='Ruta de distribución de población por categoría')
    parser.add_argument('-d', default='30m', help='Duración total de simulación')
    parser.add_argument('-v', default='5s', help='Intervalo de simulación')
    parser.add_argument('-m', type=int, default=100, help='Número de usuarios a simular')
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
        total = sum(base.values())
        cat_aff = {c: v/total for c, v in base.items()}
        sub_aff = {}
        for c, lst in subcats.items():
            for sc in lst:
                sub_aff[sc] = max(0, cat_aff[c] + np.random.normal(0, 0.1))
        tot_sc = sum(sub_aff.values())
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

def system_strategy(users, subcats, pub_dur, k=2):
    scores = {}
    for sc_list in subcats.values():
        for sc in sc_list:
            total_aff = sum(u['aff'].get(sc, 0) for u in users if u['connected'])
            scores[sc] = pub_dur * total_aff
    chosen = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{'sub': sc, 'dur': pub_dur} for sc in chosen]

def format_hms(seconds):
    return str(timedelta(seconds=int(seconds)))

def simulate(users, subcats, dur, step):
    stats = {'attention': {sc: 0 for lst in subcats.values() for sc in lst}, 'count': 0}
    sim = 0
    while sim < dur:
        pubs = system_strategy(users, subcats, pub_dur=30, k=2)
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
    return stats

def save_results(stats, cat_dist, subcats, start, end, args):
    base = Path('results')
    base.mkdir(exist_ok=True)
    existing = sorted(base.glob('prueba*'))
    idx = len(existing) + 1
    dst = base / f'prueba{idx:03d}'
    dst.mkdir()
    labels = list(subcats.keys())
    real = [cat_dist.get(c, 0) for c in labels]
    learned = [sum(stats['attention'][sc] for sc in subcats[c]) for c in labels]
    plt.figure()
    plt.bar(labels, real)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(dst / 'real_pref.jpeg')
    plt.figure()
    plt.bar(labels, learned)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(dst / 'learned_pref.jpeg')
    info = {
        'config': vars(args),
        'inicio': start.isoformat(),
        'fin': end.isoformat(),
        'duracion_simulacion': (end - start).total_seconds(),
        'total_pubs': stats['count'],
        'promedio_atencion': (sum(stats['attention'].values()) / stats['count']) if stats['count'] else 0,
        'atencion_por_sub': stats['attention'],
        'atencion_por_cat': {c: sum(stats['attention'][sc] for sc in subcats[c]) for c in subcats}
    }
    with open(dst / 'informe.json', 'w') as f:
        json.dump(info, f, indent=2)

def main():
    args = parse_args()
    dur = parse_time(args.d)
    step = parse_time(args.v)
    steps = dur / step
    expected_real = dur if step > 0 else 0
    start = datetime.now()
    print(f"Inicio de simulación: {start.isoformat()}")
    print(f"Duración ficticia: {args.d} ({dur} segundos)")
    print(f"Velocidad: {args.v} ({step} segundos por paso)")
    print(f"Número de pasos: {int(steps)}")
    print(f"Tiempo real estimado: {format_hms(expected_real)}")
    subcats = load_categories(Path('data') / 'lista-categorias.csv')
    cat_dist = load_population(Path(args.p))
    users = init_users(args.m, cat_dist, subcats)
    stats = simulate(users, subcats, dur, step)
    end = datetime.now()
    print(f"Fin de simulación: {end.isoformat()}")
    print(f"Duración real: {format_hms((end - start).total_seconds())}")
    save_results(stats, cat_dist, subcats, start, end, args)

if __name__ == '__main__':
    main()
