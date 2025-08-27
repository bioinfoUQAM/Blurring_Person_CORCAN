# train_yolov8.py

from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")

    # === Requeridos / principais ===
    parser.add_argument('--model', type=str, required=True, help='YOLOv8 model path or pretrained weights')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')

    # === Otimizador & LR ===
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use: SGD, Adam, AdamW')
    parser.add_argument('--lr0', type=float, default=None, help='Initial learning rate (default depends on model)')
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for SGD/Adam')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--cos_lr', action='store_true', help='Use cosine learning rate scheduler')

    # === Experimento / Output ===
    parser.add_argument('--name', type=str, default='yolov8_custom', help='Experiment name')
    parser.add_argument('--project', type=str, default='runs/train', help='Project output folder')
    parser.add_argument('--device', type=str, default='0', help='CUDA device ID(s), e.g., 0 or 0,1,2,3 or cpu')

    # === Outros ===
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')


    return parser.parse_args()

def main(opt):
    model = YOLO(opt.model)

    # Monta argumentos dinamicamente
    train_args = {
        'data': opt.data,
        'epochs': opt.epochs,
        'imgsz': opt.imgsz,
        'batch': opt.batch,
        'optimizer': opt.optimizer,
        'name': opt.name,
        'project': opt.project,
        'device': opt.device,
        'patience': opt.patience,
        'verbose': opt.verbose,
    }

    # Apenas adiciona extras se fornecidos
    if opt.lr0 is not None:
        train_args['lr0'] = opt.lr0
    if opt.momentum is not None:
        train_args['momentum'] = opt.momentum
    if opt.weight_decay is not None:
        train_args['weight_decay'] = opt.weight_decay
    if opt.cos_lr:
        train_args['cos_lr'] = True

    print(f"\n🚀 Iniciando treinamento com argumentos:\n{train_args}\n")
    model.train(**train_args)

if __name__ == '__main__':
    opt = parse_args()
    main(opt)
