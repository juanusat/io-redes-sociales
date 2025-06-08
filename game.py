import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from pathlib import Path
import time

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
    parser.add_argument('-p', required=True)
    parser.add_argument('-d', default='30m')
    parser.add_argument('-v', default='5s')
    parser.add_argument('-m', type=int, default=100)
    args = parser.parse_args()
    if args.i:
        args.p = input('Ruta de poblacion: ')
        args.d = input('Duracion (e.g. 2m,4h): ') or args.d
        args.v = input('Velocidad (e.g. 10s,3m): ') or args.v
        args.m = int(input('Muestra (# usuarios): ') or args.m)
    return args

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
        total_base = sum(base.values())
        cat_aff = {c: v / total_base for c, v in base.items()}
        sub_aff = {}
        for c, lst in subcats.items():
            for sc in lst:
                sub_aff[sc] = max(0, cat_aff[c] + np.random.normal(0, 0.1))
        total_sub = sum(sub_aff.values())
        sub_aff = {k: v / total_sub for k, v in sub_aff.items()}
        users.append({'id': i, 'aff': sub_aff, 'connected': True, 'unint': 0})
    return users

def evaluate_attention(user, pub_dur, sub):
    t = pub_dur * user['aff'].get(sub, 0)
    leave = False
    if t < 5:
        user['unint'] += 1
        if user['unint'] >= 3:
            user['connected'] = False
            user['unint'] = 0
            leave = True
    return t, leave

def handle_notifications(users):
    for u in users:
        if not u['connected']:
            top = max(u['aff'], key=u['aff'].get)
            if u['aff'][top] > np.random.rand():
                u['connected'] = True

def simulate(users, cat_dist, subcats, dur, step):
    stats = {'attention': {sc: 0 for lst in subcats.values() for sc in lst}, 'count': 0}
    sim = 0
    start = datetime.now()
    while sim < dur:
        pubs = []
        for c, lst in subcats.items():
            for _ in range(2):
                pubs.append({'sub': np.random.choice(lst), 'dur': 30})
        for u in users:
            if not u['connected']:
                continue
            for pub in pubs:
                t, left = evaluate_attention(u, pub['dur'], pub['sub'])
                stats['attention'][pub['sub']] += t
                stats['count'] += 1
                if left:
                    break
        handle_notifications(users)
        sim += step
    return stats, start, datetime.now()

def save_results(stats, cat_dist, subcats, start, end, args):
    base = Path('results')
    base.mkdir(exist_ok=True)
    existing = sorted(base.glob('prueba*'))
    num = len(existing) + 1
    dst = base / f'prueba{num:03d}'
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
        'config': {'poblacion': args.p, 'duracion': args.d, 'velocidad': args.v, 'muestra': args.m},
        'inicio': start.isoformat(),
        'fin': end.isoformat(),
        'total_pubs': stats['count'],
        'promedio_atencion': stats['count'] and sum(stats['attention'].values())/stats['count'] or 0,
        'atencion_por_sub': stats['attention'],
        'atencion_por_cat': {c: sum(stats['attention'][sc] for sc in subcats[c]) for c in subcats}
    }
    with open(dst / 'informe.json', 'w') as f:
        json.dump(info, f, indent=2)

def main():
    args = parse_args()
    dur = parse_time(args.d)
    step = 30
    subcats = load_categories(Path('data') / 'lista-categorias.csv')
    cat_dist = load_population(Path(args.p))
    users = init_users(args.m, cat_dist, subcats)
    stats, start, end = simulate(users, cat_dist, subcats, dur, step)
    save_results(stats, cat_dist, subcats, start, end, args)

if __name__ == '__main__':
    main()