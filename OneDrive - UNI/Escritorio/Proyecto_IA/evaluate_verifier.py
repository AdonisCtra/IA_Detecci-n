import argparse
import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import cv2

from classifier import Verifier


def load_dataset_images(root: str):
    # expects root/phone and root/not_phone
    imgs = []
    labels = []
    for label_name, lab in [('not_phone', 0), ('phone', 1)]:
        p = Path(root) / label_name
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                imgs.append(str(f))
                labels.append(lab)
    return imgs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset', help='Carpeta con phone/ not_phone')
    parser.add_argument('--model', default='classifier.pth', help='Ruta al verificador .pth')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    imgs, labels = load_dataset_images(args.data)
    if not imgs:
        print('No se encontraron imágenes en', args.data)
        return

    v = Verifier(args.model)
    crops = []
    for p in imgs:
        im = cv2.imread(p)
        if im is None:
            im = np.zeros((224,224,3), dtype=np.uint8)
        crops.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    probs = v.predict_probs(crops, batch_size=args.batch)
    probs = np.array(probs)
    labels = np.array(labels)

    # threshold sweep to find best F1
    best = (0.5, 0.0)
    for thr in np.linspace(0.1, 0.9, 41):
        preds = (probs >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f > best[1]:
            best = (thr, f)

    optimal_thr = best[0]
    final_preds = (probs >= optimal_thr).astype(int)
    acc = accuracy_score(labels, final_preds)
    p, r, f, _ = precision_recall_fscore_support(labels, final_preds, average='binary', zero_division=0)

    print(f'Optimal threshold (F1): {optimal_thr:.2f} -> F1={best[1]:.3f}')
    print(f'Accuracy={acc:.3f} Precision={p:.3f} Recall={r:.3f} F1={f:.3f}')


if __name__ == '__main__':
    main()
