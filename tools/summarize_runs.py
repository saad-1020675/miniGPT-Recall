"""Rebuild a machine-readable index of supplied training diagnostics."""
import argparse
import json
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    valid = [r for r in rows if r.get('validation',{}).get('conditions',{}).get('Normal')]
    if not valid:
        raise ValueError(f'No Normal validation rows in {path}')
    best = max(valid, key=lambda r:r['validation']['conditions']['Normal']['exact_accuracy'])
    first = max(valid, key=lambda r:r['validation']['conditions']['Normal']['first_accuracy'])
    result = dict(file=path.name, records=len(rows), first_logged_tokens=rows[0]['tokens'],
        final_logged_tokens=rows[-1]['tokens'],
        best_exact=best['validation']['conditions']['Normal']['exact_accuracy'],
        best_exact_at_tokens=best['tokens'],
        best_first=first['validation']['conditions']['Normal']['first_accuracy'],
        best_first_at_tokens=first['tokens'],
        final_normal=valid[-1]['validation']['conditions']['Normal'],
        final_no_read=valid[-1]['validation']['conditions'].get('NoRead'),
        writer_config=rows[-1].get('writer_config'),
        recall_includes_newline=rows[-1].get('recall_includes_newline'),
        support_window=rows[-1].get('support_window','not recorded'),
        support_label=rows[-1].get('support_label','not recorded'),
        recall_ratio=rows[-1].get('recall_ratio'),
        ratio_basis=rows[-1].get('ratio_basis'))
    prose = [r for r in rows if r.get('prose_validation_loss') is not None]
    if prose:
        p = min(prose,key=lambda r:r['prose_validation_loss'])
        result.update(best_prose_lm=p['prose_validation_loss'],best_prose_at_tokens=p['tokens'],
            recall_at_best_prose=p['validation']['conditions']['Normal']['exact_accuracy'],
            final_prose_lm=prose[-1]['prose_validation_loss'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('results/training'))
    p.add_argument('--output', type=Path, default=Path('results/training_index.json'))
    args=p.parse_args()
    paths=sorted(args.input.glob('*.jsonl'))
    if not paths:raise ValueError('No training JSONL files')
    result=dict(note='Historical logs; peaks may be different checkpoints; missing metadata is unknown, not inferred from filenames.',
                runs=[summarize(path) for path in paths])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'Indexed {len(paths)} training logs.')


if __name__=='__main__':
    main()
