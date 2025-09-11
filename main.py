# main.py
import argparse
from src.train import train_model
from src.train_cnn import train_cnn_model
from src.evaluate import evaluate_mfcc_model, evaluate_cnn_model
from src.predict import predict_file_mfcc, predict_file_cnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎵 Music Genre Classification Project 🎵")
    parser.add_argument("--mode", choices=["mfcc", "cnn"], default="mfcc",
                        help="نوع مدل: mfcc یا cnn (پیش‌فرض: mfcc)")
    parser.add_argument("--predict", type=str, default=None,
                        help="مسیر یک فایل صوتی برای تست پیش‌بینی")
    args = parser.parse_args()

    print("🎵 Music Genre Classification Project 🎵")
    print(f"👉 Running in {args.mode.upper()} mode")

    if args.mode == "mfcc":
        model, history = train_model()
        evaluate_mfcc_model(model)
        if args.predict:
            predict_file_mfcc(model, args.predict)

    elif args.mode == "cnn":
        model, history = train_cnn_model()
        evaluate_cnn_model(model)
        if args.predict:
            predict_file_cnn(model, args.predict)
