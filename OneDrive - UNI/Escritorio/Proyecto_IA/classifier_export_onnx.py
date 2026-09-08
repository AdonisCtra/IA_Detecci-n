import argparse
import torch
from classifier import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='classifier.pth')
    parser.add_argument('--out', default='classifier.onnx')
    args = parser.parse_args()

    state = torch.load(args.model, map_location='cpu')
    model = build_model(num_classes=2)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, args.out, opset_version=12, do_constant_folding=True, input_names=['input'], output_names=['output'])
    print('Exportado ONNX a', args.out)


if __name__ == '__main__':
    main()
